param(
    [string]$RepoName = "edge2product-pix2pix",
    [string]$Description = "Edge2Product Pix2Pix sketch-to-product image generation project",
    [string]$GhPath = "",
    [switch]$Private
)

$ErrorActionPreference = "Stop"

function Resolve-Gh {
    param([string]$ExplicitPath)

    $candidates = @()
    if ($ExplicitPath) {
        $candidates += $ExplicitPath
    }

    $cmd = Get-Command gh -ErrorAction SilentlyContinue
    if ($cmd) {
        $candidates += $cmd.Source
    }

    $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    $portable = Join-Path $projectRoot "..\.tools\gh\extracted\bin\gh.exe"
    $candidates += $portable

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw "GitHub CLI was not found. Install gh or pass -GhPath."
}

function Invoke-Gh {
    param(
        [string]$GhExe,
        [string[]]$GhArguments,
        [string]$InputJson = $null,
        [switch]$AllowFailure
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($null -eq $InputJson) {
            $output = & $GhExe @GhArguments 2>&1
        } else {
            $output = $InputJson | & $GhExe @GhArguments 2>&1
        }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    $text = ($output | ForEach-Object { "$_" }) -join "`n"

    if ($code -ne 0 -and -not $AllowFailure) {
        throw "gh $($GhArguments -join ' ') failed:`n$text"
    }

    [pscustomobject]@{
        Code = $code
        Text = $text
    }
}

function Convert-ToUnixPath {
    param([string]$Path)
    $Path.Replace("\", "/")
}

function Get-ProjectRelativePath {
    param(
        [string]$BasePath,
        [string]$FullPath
    )

    $base = (Resolve-Path -LiteralPath $BasePath).Path.TrimEnd("\") + "\"
    $full = (Resolve-Path -LiteralPath $FullPath).Path
    $baseUri = New-Object System.Uri($base)
    $fullUri = New-Object System.Uri($full)
    [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($fullUri).ToString()).Replace("/", "\")
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$GhExe = Resolve-Gh -ExplicitPath $GhPath

$auth = Invoke-Gh -GhExe $GhExe -GhArguments @("auth", "status") -AllowFailure
if ($auth.Code -ne 0) {
    Write-Host "GitHub CLI is not logged in. Run gh auth login, then rerun scripts/publish_to_github_api.ps1."
    exit 1
}

$owner = (Invoke-Gh -GhExe $GhExe -GhArguments @("api", "user", "--jq", ".login")).Text.Trim()
if (-not $owner) {
    throw "Could not determine authenticated GitHub user."
}

$repoCheck = Invoke-Gh -GhExe $GhExe -GhArguments @("api", "repos/$owner/$RepoName") -AllowFailure
if ($repoCheck.Code -eq 0) {
    Write-Host "Repository already exists: https://github.com/$owner/$RepoName"
} else {
    $repoPayload = @{
        name = $RepoName
        description = $Description
        private = [bool]$Private
        auto_init = $false
    } | ConvertTo-Json
    Invoke-Gh -GhExe $GhExe -GhArguments @("api", "user/repos", "--method", "POST", "--input", "-") -InputJson $repoPayload | Out-Null
    Write-Host "Created repository: https://github.com/$owner/$RepoName"
}

$mainRefCheck = Invoke-Gh -GhExe $GhExe -GhArguments @("api", "repos/$owner/$RepoName/git/ref/heads/main") -AllowFailure
if ($mainRefCheck.Code -ne 0) {
    Write-Host "Initializing empty repository branch ..."
    $bootstrapContent = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes("bootstrap`n"))
    $bootstrapPayload = @{
        message = "Bootstrap repository"
        content = $bootstrapContent
        branch = "main"
    } | ConvertTo-Json
    Invoke-Gh -GhExe $GhExe -GhArguments @("api", "repos/$owner/$RepoName/contents/.github-upload-bootstrap", "--method", "PUT", "--input", "-") -InputJson $bootstrapPayload | Out-Null
}

$topFiles = @(
    ".gitignore",
    "README.md",
    "requirements.txt",
    "train.py",
    "infer.py",
    "evaluate.py",
    "demo_gradio.py",
    "make_subset.py",
    "run_experiment.py"
)

$includeRoots = @(
    "configs",
    "models",
    "datasets",
    "utils",
    "scripts",
    "report",
    "assets",
    "data\edges2shoes_100",
    "outputs\edge2shoes_100\logs",
    "outputs\edge2shoes_100\curves",
    "outputs\edge2shoes_100\metrics",
    "outputs\edge2shoes_100\samples",
    "outputs\edge2shoes_100\inference"
)

$files = New-Object System.Collections.Generic.List[System.IO.FileInfo]

foreach ($file in $topFiles) {
    $path = Join-Path $ProjectRoot $file
    if (Test-Path -LiteralPath $path) {
        $files.Add((Get-Item -LiteralPath $path))
    }
}

foreach ($root in $includeRoots) {
    $path = Join-Path $ProjectRoot $root
    if (Test-Path -LiteralPath $path) {
        Get-ChildItem -LiteralPath $path -Recurse -File | ForEach-Object { $files.Add($_) }
    }
}

$maxRegularFileBytes = 100MB
$filtered = $files |
    Sort-Object FullName -Unique |
    Where-Object {
        $relative = Convert-ToUnixPath (Get-ProjectRelativePath -BasePath $ProjectRoot -FullPath $_.FullName)
        $include = $true
        if ($relative -match "^docs/") { $include = $false }
        if ($relative -match "^github_upload/") { $include = $false }
        if ($relative -match "^\.tools/") { $include = $false }
        if ($relative -match "^\.venv/") { $include = $false }
        if ($relative -match "^\.idea/") { $include = $false }
        if ($relative -match "/checkpoints/") { $include = $false }
        if ($relative -match "\.(pth|pt|ckpt|onnx)$") { $include = $false }
        if ($relative -match "^data/edges2shoes(/|$)") { $include = $false }
        if ($relative -eq "data/edges2shoes.tar.gz") { $include = $false }
        if ($relative -match "^report/.*\.(aux|bbl|blg|fdb_latexmk|fls|out|synctex\.gz|toc|xdv)$") { $include = $false }
        if ($_.Length -gt $maxRegularFileBytes) {
            throw "Refusing to upload $relative because it is larger than 100 MiB."
        }
        $include
    }

if (-not $filtered -or $filtered.Count -eq 0) {
    throw "No files selected for upload."
}

Write-Host "Uploading $($filtered.Count) files to https://github.com/$owner/$RepoName ..."

$treeEntries = New-Object System.Collections.Generic.List[object]
$index = 0
foreach ($file in $filtered) {
    $index += 1
    $relative = Convert-ToUnixPath (Get-ProjectRelativePath -BasePath $ProjectRoot -FullPath $file.FullName)
    Write-Host ("[{0}/{1}] {2}" -f $index, $filtered.Count, $relative)

    $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
    $content = [Convert]::ToBase64String($bytes)
    $blobPayload = @{
        content = $content
        encoding = "base64"
    } | ConvertTo-Json

    $blobText = (Invoke-Gh -GhExe $GhExe -GhArguments @("api", "repos/$owner/$RepoName/git/blobs", "--method", "POST", "--input", "-") -InputJson $blobPayload).Text
    $blob = $blobText | ConvertFrom-Json
    $mode = "100644"
    if ($relative -match "^scripts/.*\.sh$") {
        $mode = "100755"
    }

    $treeEntries.Add([pscustomobject]@{
        path = $relative
        mode = $mode
        type = "blob"
        sha = $blob.sha
    })
}

$treePayload = @{
    tree = $treeEntries
} | ConvertTo-Json -Depth 8

$treeText = (Invoke-Gh -GhExe $GhExe -GhArguments @("api", "repos/$owner/$RepoName/git/trees", "--method", "POST", "--input", "-") -InputJson $treePayload).Text
$tree = $treeText | ConvertFrom-Json

$ref = Invoke-Gh -GhExe $GhExe -GhArguments @("api", "repos/$owner/$RepoName/git/ref/heads/main") -AllowFailure
$parents = @()
if ($ref.Code -eq 0) {
    $refObj = $ref.Text | ConvertFrom-Json
    $parents = @($refObj.object.sha)
}

$commitPayloadTable = @{
    message = "Initial commit: Edge2Product Pix2Pix GAN project"
    tree = $tree.sha
}
if ($parents.Count -gt 0) {
    $commitPayloadTable.parents = $parents
}

$commitPayload = $commitPayloadTable | ConvertTo-Json -Depth 6
$commitText = (Invoke-Gh -GhExe $GhExe -GhArguments @("api", "repos/$owner/$RepoName/git/commits", "--method", "POST", "--input", "-") -InputJson $commitPayload).Text
$commit = $commitText | ConvertFrom-Json

if ($parents.Count -gt 0) {
    $updatePayload = @{
        sha = $commit.sha
        force = $false
    } | ConvertTo-Json
    Invoke-Gh -GhExe $GhExe -GhArguments @("api", "repos/$owner/$RepoName/git/refs/heads/main", "--method", "PATCH", "--input", "-") -InputJson $updatePayload | Out-Null
} else {
    $refPayload = @{
        ref = "refs/heads/main"
        sha = $commit.sha
    } | ConvertTo-Json
    Invoke-Gh -GhExe $GhExe -GhArguments @("api", "repos/$owner/$RepoName/git/refs", "--method", "POST", "--input", "-") -InputJson $refPayload | Out-Null
    $defaultPayload = @{
        default_branch = "main"
    } | ConvertTo-Json
    Invoke-Gh -GhExe $GhExe -GhArguments @("api", "repos/$owner/$RepoName", "--method", "PATCH", "--input", "-") -InputJson $defaultPayload | Out-Null
}

Write-Host ""
Write-Host "Upload complete."
Write-Host "GitHub URL: https://github.com/$owner/$RepoName"
Write-Host "Commit: $($commit.sha)"
