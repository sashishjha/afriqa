# Lessons

### Do not auto-run cluster commands
- **Context**: Any task involving Slurm scripts, deployment scripts, or remote cluster commands (like `sbatch`, `srun`, etc.).
- **Mistake**: Attempted to automatically run `sbatch` on the local machine assuming it had cluster access. The user's environment separates local file editing from remote cluster execution (which requires manual syncing).
- **Rule**: Never run cluster submission commands (e.g., `sbatch`) automatically. Only prepare and save the files locally, then explicitly ask the user to sync them and run the command on their remote cluster terminal.
- **Date**: 2026-06-17
