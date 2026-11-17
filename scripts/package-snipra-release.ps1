param(
  [string]$OutputDir = "dist-release"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
# La version sale del pyproject, igual que en package-snipra-release.sh, para que
# los dos artefactos del mismo tag se llamen parejo y puedan convivir en un solo
# release de GitHub (snipra-windows-X.Y.Z.zip / snipra-macos-X.Y.Z.zip).
$version = (Select-String -Path (Join-Path $root "backend\pyproject.toml") `
  -Pattern '^version = "(.*)"').Matches[0].Groups[1].Value
if (-not $version) { $version = "0.0.0" }
$outRoot = Join-Path $root $OutputDir
$stage = Join-Path $outRoot "snipra"
$zip = Join-Path $outRoot "snipra-windows-$version.zip"

# Validate the staging boundary before any recursive removal.
$rootPath = [IO.Path]::GetFullPath($root.Path).TrimEnd('\') + '\'
$stagePath = [IO.Path]::GetFullPath($stage)
if (-not $stagePath.StartsWith($rootPath, [StringComparison]::OrdinalIgnoreCase)) {
  throw "El staging debe permanecer dentro del checkout."
}

# Skip runtime trees before copying: copying several GB of installed vision
# dependencies only to delete them makes a clean-clone package needlessly slow.
$ReleaseFiles = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)
$trackedFiles = & git -C $root -c core.quotepath=false ls-files
if ($LASTEXITCODE -ne 0) { throw 'El empaquetado requiere un checkout Git verificable.' }
foreach ($trackedFile in $trackedFiles) { [void]$ReleaseFiles.Add($trackedFile) }

function Copy-ReleaseItem {
  param([string]$Source, [string]$Destination)
  $entry = Get-Item -LiteralPath $Source -Force
  if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { return }
  if ($entry.PSIsContainer) {
    if ($entry.Name -in @('.venv', 'venv', 'node_modules', '.git', 'storage',
        '__pycache__', '.pytest_cache', '.ruff_cache', '.vite', 'logs', 'cache',
        '.cache', 'temp', 'tmp', 'artifacts', 'coverage', 'dist', 'build')) { return }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    foreach ($child in Get-ChildItem -LiteralPath $Source -Force) {
      Copy-ReleaseItem -Source $child.FullName -Destination (Join-Path $Destination $child.Name)
    }
  } else {
    $relative = $entry.FullName.Substring($rootPath.Length).Replace('\', '/')
    if (-not $ReleaseFiles.Contains($relative)) { return }
    if ($entry.Name -eq '.env' -or ($entry.Name -like '.env.*' -and $entry.Name -ne '.env.example') -or
        $entry.Name -eq 'Cookies' -or $entry.Name -eq 'cookies.txt' -or
        $entry.Name -like '*.cookies.*' -or $entry.Extension -in @('.db', '.sqlite', '.sqlite3',
        '.log', '.tmp', '.temp', '.bak', '.mp4', '.mkv', '.mov', '.webm', '.mp3', '.wav',
        '.m4a', '.flac', '.opus')) { return }
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
  }
}

# Compress-Archive de PowerShell 5.1 guarda las rutas con "\", contra la especificacion
# ZIP (APPNOTE 4.4.17.1 exige "/"). Windows lo tolera, pero unzip avisa y los extractores
# estrictos dejan archivos planos llamados "extension\popup.js" en vez de la carpeta.
function New-ZipFromDirectory {
  param([string]$SourceDir, [string]$Destination)

  Add-Type -AssemblyName System.IO.Compression | Out-Null
  Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null

  if (Test-Path -LiteralPath $Destination) {
    Remove-Item -LiteralPath $Destination -Force
  }
  $prefix = (Resolve-Path $SourceDir).Path.TrimEnd('\') + '\'
  $archive = [System.IO.Compression.ZipFile]::Open(
    $Destination, [System.IO.Compression.ZipArchiveMode]::Create)
  try {
    foreach ($file in Get-ChildItem -LiteralPath $SourceDir -Recurse -File -Force) {
      $entryName = $file.FullName.Substring($prefix.Length).Replace('\', '/')
      [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
        $archive, $file.FullName, $entryName,
        [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
  } finally {
    $archive.Dispose()
  }
}

if (Test-Path $stage) {
  Remove-Item -LiteralPath $stage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stage | Out-Null

$include = @(
  "backend",
  "frontend",
  "scripts",
  "extension",
  "Snipra.app",
  # Lanzador sin consola de Windows (equivalente a Snipra.app). Sin el, el zip
  # solo trae el .bat, que abre una ventana de cmd y deja un "pause" al final.
  "Snipra.exe",
  "Start-Snipra.bat",
  "Start-Snipra.ps1",
  "Start-Snipra.command",
  "Stop-Snipra.bat",
  "README.md",
  "LICENSE",
  "CHANGELOG.md",
  "docs",
  "snipra.ico",
  "snipra.icns"
)

foreach ($item in $include) {
  $src = Join-Path $root $item
  if (Test-Path $src) {
    Copy-ReleaseItem -Source $src -Destination (Join-Path $stage $item)
  }
}

# El Agent Runtime necesita el Skill canonico en tiempo de ejecucion
# (backend/app/editor/director.py resuelve SKILL_ROOT = PROJECT_DIR/.agents/skills/
# snipra-video-director). No se copia todo .agents/ -- solo esta carpeta, para no
# arrastrar otros skills sueltos (p.ej. snipra-editorial-director2.zip). Fail closed:
# si el Skill canonico no existe en el repo, no tiene sentido seguir empaquetando.
$skillSrc = Join-Path $root ".agents\skills\snipra-video-director"
if (-not (Test-Path (Join-Path $skillSrc "SKILL.md"))) {
  throw "Falta el Skill canonico .agents\skills\snipra-video-director\SKILL.md en el repo -- abortando empaquetado."
}
$skillDestParent = Join-Path $stage ".agents\skills"
New-Item -ItemType Directory -Force -Path $skillDestParent | Out-Null
Copy-ReleaseItem -Source $skillSrc -Destination (Join-Path $skillDestParent 'snipra-video-director')
if (-not (Test-Path (Join-Path $stage ".agents\skills\snipra-video-director\SKILL.md"))) {
  throw "SKILL.md no llego al staging tras copiar el Skill canonico -- abortando empaquetado."
}

# backend\.env lleva las API keys y backend\cookies.txt la sesion real del
# navegador (YT_DLP_COOKIES_PATH): ninguno de los dos puede viajar en un release.
$remove = @(
  "backend\.env",
  "backend\cookies.txt",
  "backend\.venv",
  "frontend\node_modules",
  "extension\node_modules",
  "frontend\dist",
  "frontend\.vite",
  "storage",
  ".git"
)

foreach ($item in $remove) {
  $target = Join-Path $stage $item
  if (Test-Path $target) {
    Remove-Item -LiteralPath $target -Recurse -Force
  }
}

# Caches locales de pytest/ruff pueden aparecer en cualquier subcarpeta (no solo
# backend\), por ejemplo si pytest se corrio alguna vez con cwd=frontend por error.
# Igual que __pycache__, se limpian por nombre en todo el arbol en vez de una
# ruta fija.
foreach ($cacheDir in @("__pycache__", ".pytest_cache", ".ruff_cache")) {
  Get-ChildItem -LiteralPath $stage -Directory -Recurse -Filter $cacheDir -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }
}

New-Item -ItemType Directory -Force -Path (Join-Path $stage "storage") | Out-Null

# Guarda de regresion: evita repetir el bug de v1.3.0 (Agent Runtime empaquetado sin
# Skill). Corre justo antes de zipear, sobre el staging ya armado.
$stagedSkillsDir = Join-Path $stage ".agents\skills"
$stagedSkillDir = Join-Path $stagedSkillsDir "snipra-video-director"
if (-not (Test-Path (Join-Path $stagedSkillDir "SKILL.md"))) {
  throw "Guarda de empaquetado: falta SKILL.md canonico en el staging -- abortando."
}
# Referencias que backend/app/editor/director.py lee por nombre (CORE_REFERENCES +
# todos los valores de INTENT_REFERENCES). Si falta alguna, el Agent Runtime falla
# en tiempo de ejecucion al pedir un intent que la necesite.
$requiredReferences = @(
  "editorial-core.md", "execution-safety.md", "verification.md",
  "hook-director.md", "event-model.md", "best-event-pool.md",
  "highlight-director.md", "event-coherence.md",
  "montage-director.md",
  "continuity.md", "pacing.md",
  "captions.md",
  "audio.md",
  "b-roll.md"
)
foreach ($ref in $requiredReferences) {
  $refPath = Join-Path $stagedSkillDir "references\$ref"
  if (-not (Test-Path $refPath)) {
    throw "Guarda de empaquetado: falta la referencia requerida references\$ref del Skill canonico -- abortando."
  }
}
if (Test-Path (Join-Path $stagedSkillsDir "snipra-editorial-director2.zip")) {
  throw "Guarda de empaquetado: snipra-editorial-director2.zip no debe viajar en el release -- abortando."
}
$stagedAgentsEntries = Get-ChildItem -LiteralPath (Join-Path $stage ".agents") -Force
if (($stagedAgentsEntries | Measure-Object).Count -ne 1 -or $stagedAgentsEntries[0].Name -ne "skills") {
  throw "Guarda de empaquetado: .agents en staging trae algo distinto de skills\ -- abortando."
}
$stagedSkillsEntries = Get-ChildItem -LiteralPath $stagedSkillsDir -Force
if (($stagedSkillsEntries | Measure-Object).Count -ne 1 -or $stagedSkillsEntries[0].Name -ne "snipra-video-director") {
  throw "Guarda de empaquetado: .agents\skills en staging trae algo distinto de snipra-video-director\ -- abortando."
}

New-ZipFromDirectory -SourceDir $stage -Destination $zip
if (-not (Test-Path $zip)) {
  throw "No se genero $zip"
}
$sizeMb = [math]::Round((Get-Item -LiteralPath $zip).Length / 1MB, 1)
Write-Host "Release package created: $zip ($sizeMb MB)"
$zipHash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
$exeHash = (Get-FileHash -LiteralPath (Join-Path $stage 'Snipra.exe') -Algorithm SHA256).Hash.ToLowerInvariant()
@("$zipHash  $([IO.Path]::GetFileName($zip))", "$exeHash  snipra/Snipra.exe") |
  Set-Content -LiteralPath (Join-Path $outRoot 'SHA256SUMS') -Encoding ASCII
