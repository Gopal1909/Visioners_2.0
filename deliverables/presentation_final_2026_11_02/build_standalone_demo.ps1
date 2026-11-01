$ErrorActionPreference = 'Stop'
$demo = Join-Path $PSScriptRoot 'demo'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$manifest = Get-Content -Raw (Join-Path $demo 'asset_manifest.json') | ConvertFrom-Json
foreach ($entry in $manifest.PSObject.Properties) {
    $path = Join-Path $demo ($entry.Name.Replace('/', '\'))
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) { throw "Asset hash mismatch: $path" }
}
$entries = Get-ChildItem (Join-Path $demo 'assets') -Filter '*.json' -File | Sort-Object Name | ForEach-Object {
    $json = [System.IO.File]::ReadAllText($_.FullName)
    $null = $json | ConvertFrom-Json
    $json = $json.Replace('<', '\u003c').Replace('>', '\u003e').Replace('&', '\u0026')
    '"assets/' + $_.Name + '":' + $json
}
$bundle = '<script>window.offlineAssets={' + ($entries -join ',') + '};</script>'
$html = [System.IO.File]::ReadAllText((Join-Path $demo 'index.html'))
$css = [System.IO.File]::ReadAllText((Join-Path $demo 'style.css'))
$js = [System.IO.File]::ReadAllText((Join-Path $demo 'viewer.js'))
$start = $js.IndexOf('async function getJSON(url)')
$end = $js.IndexOf("`n", $start)
if ($start -lt 0 -or $end -lt 0) { throw 'Cannot locate JSON loader.' }
$js = $js.Substring(0, $start) + 'async function getJSON(url) { const value = window.offlineAssets[url]; if (!value) throw Error(`Missing embedded asset: ${url}`); return value; }' + $js.Substring($end)
$html = $html.Replace('<link rel="stylesheet" href="style.css"><script defer src="viewer.js"></script>', '<style>' + $css + '</style>')
$html = $html.Replace('local-only HTTP', 'standalone file - no server')
$html = $html.Replace('href="assets/provenance.json"', 'id="provenance-download" download="provenance.json" href="#"')
$provenance = 'const link=document.getElementById("provenance-download"); link.href=URL.createObjectURL(new Blob([JSON.stringify(window.offlineAssets["assets/provenance.json"],null,2)],{type:"application/json"}));'
$html = $html.Replace('</body>', $bundle + '<script>' + $js + "`n" + $provenance + '</script></body>')
$out = Join-Path $PSScriptRoot 'CV5570_Offline_Demo.html'
[System.IO.File]::WriteAllText($out, $html, $utf8)
Get-Item -LiteralPath $out | Select-Object FullName, Length
