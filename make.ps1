param (
    [string]$Task
)

switch ($Task) {
    "embed" {
        python src/embed_database.py
    }
    "infer" {
        python src/inference.py
    }
    "test" {
        pytest tests/
    }
    default {
        Write-Host "Unknown task: $Task"
    }
}
