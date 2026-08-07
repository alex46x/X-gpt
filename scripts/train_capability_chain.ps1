[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$BaseCheckpoint = "artifacts/runs/english-chat-corrected-cpu-v1/checkpoint.pt",
    [Parameter(Mandatory = $false)]
    [string]$OutputRoot = "artifacts/runs/genesis-capabilities-v1",
    [Parameter(Mandatory = $false)]
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$base = (Resolve-Path -LiteralPath $BaseCheckpoint).Path
$stageConversation = Join-Path $OutputRoot "conversation"
$stageCode = Join-Path $OutputRoot "coding"
if (Test-Path -LiteralPath $stageConversation) { throw "Conversation stage already exists: $stageConversation" }
if (Test-Path -LiteralPath $stageCode) { throw "Coding stage already exists: $stageCode" }

$common = @(
    "--preprocessing-config", "configs/preprocessing/default.yaml",
    "--tokenizer-config", "configs/tokenizer/default.yaml",
    "--model-config", "configs/model/default.yaml",
    "--evaluation-config", "configs/evaluation/default.yaml",
    "--device", $Device
)

& py -m uv run --locked genesis-train @common `
    --dataset-config configs/dataset/english-conversations-mixed.yaml `
    --training-config configs/training/english-conversations-mixed-cpu.yaml `
    --output $stageConversation `
    --resume-from $base `
    --source-revision "continual-conversations-v1" `
    --training-run-id "genesis-capabilities-conversation-v1"
if ($LASTEXITCODE -ne 0) { throw "Conversation continuation failed with exit code $LASTEXITCODE" }

$conversationCheckpoint = (Resolve-Path -LiteralPath (Join-Path $stageConversation "checkpoint.pt")).Path
& py -m uv run --locked genesis-train @common `
    --dataset-config configs/dataset/codesearchnet-python.yaml `
    --training-config configs/training/codesearchnet-cpu.yaml `
    --output $stageCode `
    --resume-from $conversationCheckpoint `
    --source-revision "continual-codesearnet-v1" `
    --training-run-id "genesis-capabilities-coding-v1"
if ($LASTEXITCODE -ne 0) { throw "Coding continuation failed with exit code $LASTEXITCODE" }

Write-Output "Final combined bundle: $(Join-Path $stageCode 'bundle')"