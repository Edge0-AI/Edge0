<#
.SYNOPSIS
    Enables the 'Lock pages in memory' privilege (SeLockMemoryPrivilege) on Windows.
.DESCRIPTION
    Strata keeps the experts the GPU does not hold in a host memory arena. With SeLockMemoryPrivilege
    granted, Windows lets it allocate that arena in 2 MB large pages (MEM_LARGE_PAGES); without it the
    engine says at start that the large pages were refused and uses 4 KB pages (docs/DETAILS.md, #1412).
    The same can be done by hand with secpol.msc (User Rights Assignment, "Lock pages in memory").

    NOTE: Windows applies security token privilege changes ONLY after sign-out/sign-in or reboot.
#>

[CmdletBinding()]
param (
    [string]$AccountName = $env:USERNAME
)

function Test-IsAdmin {
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdmin)) {
    Write-Host "[!] This script must be run as Administrator." -ForegroundColor Yellow
    Write-Host "    Right-click PowerShell -> 'Run as Administrator', then run:"
    Write-Host "    .\tools\windows\enable_large_pages.ps1"
    Write-Host ""
    Write-Host "    GUI Alternative:"
    Write-Host "    1. Press Win + R, type 'secpol.msc' and press Enter."
    Write-Host "    2. Navigate to: Security Settings -> Local Policies -> User Rights Assignment"
    Write-Host "    3. Double-click 'Lock pages in memory'."
    Write-Host "    4. Click 'Add User or Group...', add '$AccountName', and click OK."
    Write-Host "    5. Sign out of Windows and sign back in."
    exit 1
}

Write-Host "=== Configuring 'Lock pages in memory' (SeLockMemoryPrivilege) for $AccountName ===" -ForegroundColor Cyan

$tempInf = [System.IO.Path]::GetTempFileName()
$tempSdb = [System.IO.Path]::ChangeExtension($tempInf, ".sdb")

try {
    Write-Host "1. Exporting current security policy..."
    secedit /export /cfg $tempInf | Out-Null

    $content = Get-Content $tempInf -Raw
    $userSid = (New-Object System.Security.Principal.NTAccount($AccountName)).Translate([System.Security.Principal.SecurityIdentifier]).Value

    if ($content -match "SeLockMemoryPrivilege\s*=\s*(.*)") {
        $currentList = $matches[1].Trim()
        if ($currentList -match [regex]::Escape($userSid) -or $currentList -match [regex]::Escape($AccountName)) {
            Write-Host "[OK] $AccountName already possesses SeLockMemoryPrivilege." -ForegroundColor Green
            Write-Host "     If 'whoami /priv' still shows it as missing, please SIGN OUT and sign back in to refresh your token."
            return
        }
        $newList = "$currentList,*$userSid"
        $content = $content -replace "SeLockMemoryPrivilege\s*=.*", "SeLockMemoryPrivilege = $newList"
    } else {
        # Section [Privilege Rights]
        if ($content -match "\[Privilege Rights\]") {
            $content = $content -replace "\[Privilege Rights\]", "[Privilege Rights]`r`nSeLockMemoryPrivilege = *$userSid"
        } else {
            $content += "`r`n[Privilege Rights]`r`nSeLockMemoryPrivilege = *$userSid`r`n"
        }
    }

    Set-Content -Path $tempInf -Value $content -Encoding ASCII

    Write-Host "2. Importing updated policy database..."
    secedit /import /db $tempSdb /cfg $tempInf | Out-Null

    Write-Host "3. Applying updated policy to Windows system..."
    secedit /configure /db $tempSdb /cfg $tempInf /areas USER_RIGHTS | Out-Null   # only the user rights, nothing else of the exported policy

    Write-Host ""
    Write-Host "[SUCCESS] SeLockMemoryPrivilege has been granted to $AccountName ($userSid)!" -ForegroundColor Green
    Write-Host "==========================================================================" -ForegroundColor Cyan
    Write-Host "IMPORTANT: You MUST sign out of Windows and sign back in (or restart)" -ForegroundColor Yellow
    Write-Host "           for the new privilege to take effect on your user session." -ForegroundColor Yellow
    Write-Host "==========================================================================" -ForegroundColor Cyan
}
catch {
    Write-Host "[ERROR] Failed to configure privilege automatically: $_" -ForegroundColor Red
    Write-Host "Please use the GUI steps via secpol.msc outlined above."
}
finally {
    if (Test-Path $tempInf) { Remove-Item $tempInf -Force -ErrorAction SilentlyContinue }
    if (Test-Path $tempSdb) { Remove-Item $tempSdb -Force -ErrorAction SilentlyContinue }
}
