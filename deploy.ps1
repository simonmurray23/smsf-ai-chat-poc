Param(
  [string]$FunctionName = $env:FUNCTION_NAME,
  [string]$Region       = $env:AWS_REGION,
  [switch]$Publish      # add -Publish if you want to create a new numbered version
)

$ErrorActionPreference = "Stop"

# 1) Move to the folder this script lives in (repo root)
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)

# 2) Choose what to include in the zip
#    Keep it minimal: app.py + any local packages/modules if you added them.
$paths = @("app.py")
$existing = @()
foreach ($p in $paths) { if (Test-Path $p) { $existing += $p } }

if ($existing.Count -eq 0) {
  throw "Nothing to package. Make sure app.py exists at the repo root."
}

# 3) Build ZIP (overwrites if present)
$zip = "function.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $existing -DestinationPath $zip -Force

# 4) Update the Lambda code
Write-Host "Updating Lambda $FunctionName in $Region ..."
$update = aws lambda update-function-code `
  --function-name $FunctionName `
  --zip-file fileb://$zip `
  --region $Region | ConvertFrom-Json

Write-Host ("Deployed. CodeSha256={0} LastModified={1}" -f $update.CodeSha256, $update.LastModified)

# 5) Optional: publish a new, immutable version and (optionally) move 'Prod' alias
if ($Publish) {
  $ver = (aws lambda publish-version --function-name $FunctionName --region $Region | ConvertFrom-Json).Version
  Write-Host "Published version $ver"
  # Uncomment to move 'Prod' to the new version:
  # aws lambda update-alias --function-name $FunctionName --name Prod --function-version $ver --region $Region | Out-Null
  # Write-Host "Alias 'Prod' now points to version $ver"
}

Write-Host "Done ✅"
