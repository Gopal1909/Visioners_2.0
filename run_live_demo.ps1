[CmdletBinding()]
param(
    [int]$Port = 5570,
    [switch]$Stop,
    [switch]$NoBrowser,
    [string]$Python = 'C:\Users\prani\anaconda3\envs\dl_env\python.exe'
)
$ErrorActionPreference = 'Stop'
$demo = Join-Path $PSScriptRoot 'deliverables\presentation_final_2026_11_02\demo'
$server = Join-Path $demo 'server.py'
$statePath = Join-Path $demo 'server_state.json'
$identity = 'CV5570_OFFLINE_DEMO_20261102_V1'
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Choose a port from 1024 to 65535.' }

function Get-Health([int]$Number) {
    try { return Invoke-RestMethod -Uri "http://127.0.0.1:$Number/health" -TimeoutSec 2 -UseBasicParsing } catch { return $null }
}
function Get-OwnedProcess($Saved) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$Saved.pid)" -ErrorAction SilentlyContinue
    if ($null -eq $process) { return $null }
    $expected = [System.IO.Path]::GetFullPath($server)
    if ($Saved.identity -ne $identity -or $Saved.root -ne $demo -or $Saved.server -ne $expected) { throw 'State identity/path mismatch; refusing PID action.' }
    if ([string]::IsNullOrWhiteSpace($Saved.token) -or $Saved.token -notmatch '^[a-f0-9]{32}$') { throw 'Invalid ownership token; refusing PID action.' }
    $command = [string]$process.CommandLine
    if ($command.IndexOf($expected, [StringComparison]::OrdinalIgnoreCase) -lt 0 -or $command.IndexOf([string]$Saved.token, [StringComparison]::Ordinal) -lt 0 -or $process.ExecutablePath -ne $Saved.python) { throw 'PID command/executable ownership mismatch; refusing to stop or reuse.' }
    $started = (Get-Process -Id ([int]$Saved.pid)).StartTime.ToUniversalTime().Ticks.ToString()
    if ($started -ne [string]$Saved.startTicks) { throw 'PID start time mismatch; refusing PID reuse.' }
    return $process
}
function Test-OwnedHealth($Health, $Saved) {
    return ($null -ne $Health -and $Health.identity -eq $identity -and $Health.token -eq $Saved.token -and [int]$Health.pid -eq [int]$Saved.pid -and $Health.root -eq $demo -and $Health.ready)
}
function Test-Assets([string]$BaseUrl) {
    $manifest = Get-Content -Raw (Join-Path $demo 'asset_manifest.json') | ConvertFrom-Json
    foreach ($property in $manifest.PSObject.Properties) {
        $local = Join-Path $demo ($property.Name.Replace('/', '\'))
        if (-not (Test-Path -LiteralPath $local -PathType Leaf)) { throw "Missing asset: $local" }
        if ((Get-FileHash -LiteralPath $local -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value) { throw "Asset hash mismatch: $local" }
        $response = Invoke-WebRequest -Uri ($BaseUrl + $property.Name) -Method Head -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -ne 200) { throw "Asset unavailable: $($property.Name)" }
    }
}

# Serialize independent launcher invocations; never race two children onto one port.
$mutex = New-Object System.Threading.Mutex($false, 'Local\CV5570_OFFLINE_DEMO_20261102')
$locked = $false
try {
    try { $locked = $mutex.WaitOne(10000) } catch [System.Threading.AbandonedMutexException] { $locked = $true }
    if (-not $locked) { throw 'Another demo launcher is busy. Retry shortly.' }
    $saved = $null
    if (Test-Path -LiteralPath $statePath) { $saved = Get-Content -Raw $statePath | ConvertFrom-Json }
    if ($Stop) {
        if ($null -eq $saved) { Write-Output 'No owned demo server state; no process stopped.'; return }
        $owned = Get-OwnedProcess $saved
        if ($null -ne $owned) {
            $health = Get-Health ([int]$saved.port)
            if ($null -ne $health -and -not (Test-OwnedHealth $health $saved)) { throw 'Health ownership mismatch; refusing shutdown.' }
            # Command path, executable, unique token AND creation time checked above, even if HTTP is hung.
            Stop-Process -Id ([int]$saved.pid) -ErrorAction Stop
            Wait-Process -Id ([int]$saved.pid) -Timeout 10 -ErrorAction SilentlyContinue
            Write-Output "Stopped owned demo PID $($saved.pid)."
        } else { Write-Output 'Saved demo PID no longer exists; removing stale state only.' }
        Remove-Item -LiteralPath $statePath -Force
        return
    }
    if ($null -ne $saved) {
        $owned = Get-OwnedProcess $saved
        if ($null -ne $owned) {
            if ([int]$saved.port -ne $Port) { throw "Owned demo already uses port $($saved.port). Stop it before changing ports." }
            $url = "http://127.0.0.1:$Port/"
            if (-not (Test-OwnedHealth (Get-Health $Port) $saved)) { throw 'Owned server is not ready. Run with -Stop, then restart.' }
            Test-Assets $url
            Write-Output "Reusing verified demo PID $($saved.pid): $url"
            if (-not $NoBrowser) { Start-Process $url }
            return
        }
        Remove-Item -LiteralPath $statePath -Force
    }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
    $Python = [System.IO.Path]::GetFullPath($Python)
    $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Parse('127.0.0.1'), $Port)
    try { $listener.Start() } catch { throw "Port $Port is occupied by an unowned service. No process touched; choose -Port 5571." } finally { $listener.Stop() }
    $token = [guid]::NewGuid().ToString('N')
    $outLog = Join-Path $demo 'server_stdout.log'
    $errLog = Join-Path $demo 'server_stderr.log'
    $child = Start-Process -FilePath $Python -ArgumentList @('-u', ('"' + $server + '"'), '--port', "$Port", '--token', $token) -WorkingDirectory $demo -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
    $saved = [ordered]@{ identity=$identity; pid=$child.Id; token=$token; port=$Port; root=$demo; server=[System.IO.Path]::GetFullPath($server); python=$Python; startTicks=$child.StartTime.ToUniversalTime().Ticks.ToString() }
    $saved | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
    $url = "http://127.0.0.1:$Port/"
    try {
        $ready = $false
        for ($i=0; $i -lt 40; $i++) {
            if (Test-OwnedHealth (Get-Health $Port) $saved) { $ready = $true; break }
            if ($child.HasExited) { throw "Server exited. See $errLog" }
            Start-Sleep -Milliseconds 200
        }
        if (-not $ready) { throw "Readiness timeout. See $errLog" }
        Test-Assets $url
    } catch {
        $failure = $_
        if ($null -ne (Get-OwnedProcess $saved)) { Stop-Process -Id ([int]$saved.pid) -ErrorAction SilentlyContinue }
        Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
        throw $failure
    }
    Write-Output "Ready: $url (owned PID $($saved.pid); local assets verified)."
    Write-Output 'Saved sampled predictions only. Stop safely: .\run_live_demo.ps1 -Stop'
    if (-not $NoBrowser) { Start-Process $url }
} finally {
    if ($locked) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
