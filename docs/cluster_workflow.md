# Explorer cluster workflow and VS Code Remote-SSH diagnosis

Last reviewed: 2026-07-23

## Bottom line

A reliable **local editing + remote cluster execution** workflow is practical.
It should not depend on VS Code Remote-SSH staying connected:

```text
Local Windows checkout
  edit -> commit -> push
             |
             v
          GitHub
             |
             v
Explorer login-node checkout
  clean pull -> verify commit -> sbatch
                               |
                               v
                    compute node + scratch artifacts
                               |
                               v
                scp small results/logs back locally
```

SSH should be the automation/control channel, Open OnDemand (OOD) should remain
the recovery channel, and Slurm should be the only experiment-execution channel.
VS Code Remote-SSH can be repaired and used as a convenience for browsing and
terminals, but it should not be the only way to submit, monitor, or recover a
job.

The previous Remote-SSH failure was not a basic SSH or hostname failure. Local
logs show that authentication, VS Code Server installation, and port forwarding
all succeeded. The remote VS Code Node process was then `Killed` while several
heavy extensions were being installed, followed by PTY errors and password-
prompt reconnect loops. The most likely cause is login-node resource/process
pressure triggered by the remote extension load, made less recoverable by
password-only authentication.

## Sources inspected

The following local reference was successfully opened and treated as read-only:

- `C:\Users\usp78\Desktop\on_policy_distillation`
- Git remote: `https://github.com/usp787/on_policy_distillation.git`
- Branch: `main`

High-value reference files were:

- `memory/cluster-env-setup.md`
- `memory/gpu-partition-tip.md`
- `env/setup_env.sbatch`
- `slurm/phase0_eval.sbatch`
- `slurm/phase1_grpo.sbatch`
- `slurm/phase2_distill.sbatch`
- `slurm/eval_avg4.sbatch`
- `.gitignore`
- the local-versus-cluster and environment sections of `README.md`

The current RSI repository is:

- local: `C:\Users\usp78\Desktop\RSI\RSI`
- Git remote: `https://github.com/usp787/RSI.git`
- branch: `main`

## Existing Explorer information

### Connection and storage

| Item | Existing information | Status |
| --- | --- | --- |
| Cluster | Northeastern Explorer HPC | Confirmed by reference repo and official documentation |
| SSH host | `login.explorer.northeastern.edu` | Current official hostname; use this rather than the older shorthand `explorer.northeastern.edu` |
| SSH user | `zha.j` | Confirmed by local SSH config and prior cluster notes |
| SSH port | `22` | Official default |
| OOD portal | `https://ood.explorer.northeastern.edu` | Recovery path for shell/file access |
| Remote OS | Rocky Linux, `x86_64` | Reported by the successful VS Code Server handshake |
| Home | `/home/zha.j` | Conda environment and small persistent state |
| Scratch | `/scratch/$USER` | Model caches and large experiment artifacts; subject to purge after inactivity |

The `on_policy_distillation` notes recorded a roughly 40 GB home quota and used
scratch to avoid filling it. Treat that number as a historical note and verify
the live quota before building the RSI environment.

### Known working environment from `on_policy_distillation`

This is a **June 2026 snapshot**, not a dependency lock for RSI:

- Conda environment: `$HOME/.conda/envs/opd`
- Python: 3.11
- modules: `cuda/12.8.0` and `miniconda3/25.9.1`
- environment creation used `conda-forge --override-channels` to avoid the
  non-interactive Anaconda Terms-of-Service gate
- recorded packages included PyTorch 2.8.0/cu128, Transformers 4.57.6,
  TRL 0.24.0, PEFT 0.17.1, Accelerate 1.10.1, Datasets 4.5.0, vLLM 0.11.0,
  and `math-verify`
- `flash-attn` was optional; SDPA was the fallback
- Hugging Face cache: `HF_HOME=/scratch/$USER/hf_cache`

For RSI, reuse the module/Slurm **shape**, not the `opd` environment itself.
Create a separate environment such as `$HOME/.conda/envs/rsi-restem` after the
Qwen2.5/TRL/vLLM compatibility set is selected. This prevents a working older
project from being broken by upgrades.

### Slurm conventions already proven useful

- Long single-H200 jobs used:
  `--partition=gpu --gres=gpu:h200:1 --time=08:00:00`.
- Short jobs historically used `gpu-short` or a less-contended GPU.
- Current official Explorer documentation lists H200 access through `gpu`,
  `gpu-short`, `gpu-interactive` for interactive allocation, and `multigpu`
  where authorized. An older local note also mentions `sharing`; recheck it with
  `sinfo` before using it.
- Jobs activated Conda explicitly inside each `.sbatch` file.
- Large caches, temporary files, raw generations, and checkpoints belonged on
  scratch rather than in Git or home.
- Resume-capable jobs used frequent checkpoints and `#SBATCH --open-mode=append`.
- All heavy work ran through `sbatch`; login nodes were limited to Git, file
  management, submission, and lightweight inspection.

The current Explorer guidance warns against CPU-intensive activity on login
nodes and notes that usage is monitored. A remote editor server should therefore
be kept minimal; language servers, AI assistants, indexing, tests, and model code
must not turn the login node into a compute node.

## What caused the previous Remote-SSH failure

The local VS Code logs from 2026-07-16 provide this sequence:

1. VS Code selected the correct Windows OpenSSH client at
   `C:\Windows\System32\OpenSSH\ssh.exe`.
2. Password authentication for `zha.j@login.explorer.northeastern.edu`
   succeeded.
3. VS Code Server downloaded into `/home/zha.j/.vscode-server` successfully.
4. The remote server listened on localhost, dynamic forwarding was established,
   and VS Code resolved the forwarded local port.
5. The remote side began installing multiple extensions, including GitHub Pull
   Requests, Python environment/debug support, Claude Code, and OpenAI ChatGPT.
6. The remote `code-server` Node process was then reported as `Killed`.
7. Subsequent attempts produced `Could not find pty ... on pty host`, socket
   timeouts, repeated password prompts, and eventually a canceled password
   dialog.

### Diagnosis

The evidence rules out these as the original root cause:

- incorrect Explorer hostname;
- invalid username/password;
- inability to download VS Code Server;
- unsupported remote architecture; or
- disabled SSH tunneling.

The strongest explanation is that the VS Code Server was terminated under the
load of simultaneous remote extension installation or activation on the shared
login node. The logs do not prove whether this was a per-user memory limit, an
out-of-memory kill, or an Explorer policy/process monitor, so that final
distinction requires a minimal retry or Research Computing confirmation.

Two additional reliability problems are confirmed:

- The Windows SSH config has the correct host and user but no Explorer-specific
  `IdentityFile`.
- The `.ssh` directory has no default `id_ed25519`/`id_rsa` key for Explorer, so
  VS Code relies on repeated password prompts. An unrelated PEM key should not
  be reused.

The early log messages that probe several nonexistent `ssh.exe` paths are
harmless client-discovery fallbacks: VS Code eventually finds the correct
OpenSSH binary. The Node deprecation warning is also not the crash cause.

## Recommended workflow: Git + SSH + Slurm

### Invariants

1. The local checkout is the only place where source code is edited.
2. Every cluster run uses a clean, committed Git revision.
3. The cluster checkout is pull-only for source; do not hand-edit project files
   there.
4. A dirty cluster checkout blocks submission. Do not auto-stash or auto-reset
   it.
5. Every Slurm output records the Git commit, config, environment, and dataset
   manifest used.
6. Checkpoints and raw generations stay on scratch. Only compact metrics, plots,
   and selected logs return to the local machine.
7. No training, inference, dataset preparation, verifier sweep, or evaluation is
   executed locally or on the login node.

### One-time step 1: establish key-based SSH

The current password-only path works, but it makes every reconnect fragile.
Create a dedicated Explorer key locally; do not reuse `CT-scan-key.pem`.

From local PowerShell:

```powershell
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\explorer_rsi_ed25519" -C "zha.j Explorer RSI"
```

Prefer a passphrase plus the Windows `ssh-agent`. If agent setup is not
practical, a dedicated passphrase-less key is more reliable but has a weaker
security posture; protect the Windows account and key ACLs carefully.

Copy only the `.pub` key to Explorer. This can be done through OOD shell/file
manager or, after checking the command carefully, from PowerShell:

```powershell
Get-Content "$env:USERPROFILE\.ssh\explorer_rsi_ed25519.pub" |
  ssh zha.j@login.explorer.northeastern.edu 'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys'
```

Add a new alias to the existing Windows SSH config; do not overwrite unrelated
entries:

```sshconfig
Host explorer-rsi
  HostName login.explorer.northeastern.edu
  User zha.j
  Port 22
  IdentityFile C:/Users/usp78/.ssh/explorer_rsi_ed25519
  IdentitiesOnly yes
  ServerAliveInterval 30
  ServerAliveCountMax 4
  TCPKeepAlive yes
```

Validate before opening VS Code:

```powershell
ssh -G explorer-rsi | Select-String 'hostname|user|port|identityfile'
ssh -o BatchMode=yes explorer-rsi 'printf "ssh-ok\n"; hostname; id -un'
```

The second command must complete without a password dialog. If it fails, fix
key authorization before troubleshooting VS Code.

### One-time step 2: create a clean cluster checkout

Use OOD Explorer Shell Access or normal SSH:

```bash
cd "$HOME"
test ! -e RSI || { echo "$HOME/RSI already exists; inspect it first"; exit 1; }
git clone https://github.com/usp787/RSI.git RSI
cd RSI
git switch main
mkdir -p logs
mkdir -p "/scratch/$USER/rsi/hf_cache" "/scratch/$USER/rsi/artifacts" "/scratch/$USER/rsi/tmp"
```

If the repository is private, configure read access deliberately. The cluster
does not need GitHub push credentials under the pull-only design.

Do not build the RSI environment on the login node. Add a dedicated setup
`.sbatch` after package versions are chosen, then submit it through Slurm.

### Normal edit-to-run cycle

On local Windows:

```powershell
git status --short
git add <intentional-files>
git commit -m "Describe the experiment change"
git push origin main
$localSha = (git rev-parse HEAD).Trim()
```

Synchronize and verify the remote checkout:

```powershell
ssh explorer-rsi 'cd ~/RSI && git status --short'
ssh explorer-rsi 'cd ~/RSI && git pull --ff-only'
$remoteSha = (ssh explorer-rsi 'cd ~/RSI && git rev-parse HEAD').Trim()
if ($remoteSha -ne $localSha) { throw "Cluster checkout is not at local commit $localSha" }
```

If `git status --short` prints anything on the cluster, stop and inspect it.
Generated files should be moved outside the checkout or added to `.gitignore`;
do not erase unknown changes.

Submit an implemented job from the verified revision:

```powershell
ssh explorer-rsi "cd ~/RSI && sbatch --export=ALL,CODE_COMMIT=$localSha slurm/<job>.sbatch"
```

Every RSI `.sbatch` entry point should verify the revision near its start:

```bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}"
actual_commit=$(git rev-parse HEAD)
if [[ -n "${CODE_COMMIT:-}" && "$actual_commit" != "$CODE_COMMIT" ]]; then
  echo "Commit mismatch: job=$CODE_COMMIT checkout=$actual_commit" >&2
  exit 2
fi
echo "code_commit=$actual_commit"
```

Monitor without opening the portal:

```powershell
ssh explorer-rsi 'squeue -u "$USER"'
ssh explorer-rsi 'sacct -j <job-id> --format=JobID,State,Elapsed,ExitCode,MaxRSS,AllocTRES'
ssh explorer-rsi 'tail -n 100 ~/RSI/logs/<job-log>.out'
```

Fetch only compact artifacts:

```powershell
scp explorer-rsi:/scratch/zha.j/rsi/artifacts/<run-id>/summary.json .\results\
scp explorer-rsi:/scratch/zha.j/rsi/artifacts/<run-id>/passk.png .\results\
```

OOD remains available when SSH is unavailable or when a browser-based file
manager is more convenient.

## Repairing VS Code Remote-SSH

### 1. Capture the current remote state

From OOD shell or plain SSH, run only lightweight diagnostics:

```bash
date
hostname
quota -s 2>&1 || true
df -h "$HOME" "/scratch/$USER" 2>&1 || true
du -sh ~/.vscode-server* 2>/dev/null || true
ulimit -a
ps -u "$USER" -o pid,ppid,%cpu,%mem,rss,etime,cmd | grep -E 'vscode|code-server' || true
```

If home is full, resolve storage first. Do not move the VS Code Server to
scratch by default because scratch can purge inactive data.

### 2. Reset the stale VS Code Server

First try VS Code's official command:

```text
Remote-SSH: Uninstall VS Code Server from Host...
```

This removes only the remote editor server/cache, not the project checkout. If
the command cannot stay connected, use OOD shell and make a recoverable backup:

```bash
pkill -u "$USER" -f 'vscode-server|code-server' || true
stamp=$(date +%Y%m%d-%H%M%S)
if [[ -d ~/.vscode-server ]]; then
  mv ~/.vscode-server ~/.vscode-server.backup-"$stamp"
fi
```

Delete the backup only after a clean connection has remained stable and its
contents are no longer needed.

### 3. Retry with a minimal VS Code profile

Create a VS Code profile named `Explorer Minimal`. Initially install only
Remote-SSH locally. Do not install or enable ChatGPT, Claude Code, GitHub Pull
Requests, Jupyter, Python language services, or other workspace extensions on
the remote host.

Use these local settings for the first retry:

```json
{
  "remote.SSH.remotePlatform": {
    "explorer-rsi": "linux"
  },
  "remote.SSH.showLoginTerminal": true,
  "remote.SSH.useLocalServer": false,
  "remote.SSH.defaultExtensions": []
}
```

Connect to `explorer-rsi` and keep the first window empty for several minutes.
Then open `~/RSI`, open one small file, and finally open one terminal. Add one
remote extension at a time only after the connection remains stable. Because
the intended workflow edits locally, no remote extension is required for job
submission or monitoring.

If the minimal server works but fails immediately after one extension is added,
that extension or its resource use is the isolating trigger. If the minimal
server itself is killed, stop retrying and send the evidence to Northeastern
Research Computing; this would indicate a login-node limit or policy mismatch.

If only the integrated terminal has PTY failures after a clean reset, test one
reversible setting change at a time. A useful diagnostic is setting
`"remote.SSH.useExecServer": false`, uninstalling the remote server again, and
retrying. Restore the default if it makes no difference.

## Failure-localization table

| First failing step | Likely class | Next evidence/action |
| --- | --- | --- |
| `ssh explorer-rsi` fails | DNS, network, account, host key, or authentication | Run `ssh -v explorer-rsi`; compare with OOD/status page |
| Password works but `BatchMode=yes` fails | Key not copied, wrong `IdentityFile`, or permissions | Inspect `~/.ssh/authorized_keys` and local `ssh -G explorer-rsi` |
| Server download fails | Remote egress, proxy, home quota, or permissions | Check Remote-SSH log, `quota -s`, and `df -h` |
| Server starts but forwarding fails | SSH forwarding policy or multi-login-node routing | Preserve Remote-SSH log and ask RC whether the login endpoint is load-balanced for multiple SSH connections |
| Node is `Killed` with no remote extensions | Login-node resource/policy limit | Collect quota/process/ulimit data and contact RC |
| Node is killed only during extension installation | Extension load or incompatibility | Use the minimal profile; add extensions one at a time |
| `Could not find pty` after a server crash | Stale/crashed VS Code Server state | Uninstall or move `.vscode-server`, then reconnect cleanly |
| Git pull refuses | Dirty/diverged cluster checkout or Git authentication | Stop; inspect `git status` and provenance—do not reset blindly |
| `sbatch` fails after sync | Slurm request, module, environment, or script error | Triage the first fatal Slurm/log line; this is separate from Remote-SSH |
| Job imports packages from `~/.local` | Host user-site leakage | Use an isolated environment and `PYTHONNOUSERSITE=1`; verify import paths before model work |

## Escalation package for Research Computing

If a minimal Remote-SSH server is still killed, send RC:

- exact local time and timezone of the attempt;
- username `zha.j` and the login-node hostname printed by `hostname`;
- VS Code and Remote-SSH versions;
- the Remote-SSH output from authentication through the first `Killed` line;
- output of `quota -s`, `df -h $HOME`, `du -sh ~/.vscode-server*`, and
  `ulimit -a`; and
- a statement that plain SSH succeeds and TCP forwarding/server startup were
  successful before the server process was killed.

Do not send private keys, passwords, access tokens, full environment dumps, or
unredacted credential files. Explorer support is available through
`rchelp@northeastern.edu` and the RC service portal.

## Recommended order for this project

1. Set up and verify the dedicated SSH key.
2. Establish the clean `~/RSI` pull-only checkout.
3. Prove Git sync, commit verification, `squeue`, and log retrieval with no
   experiment execution.
4. Repair Remote-SSH with the minimal profile; keep it optional.
5. Use the isolated `rsi-restem` environment and Slurm scripts now staged in
   this repository; follow `docs/experiment_runbook.md`.
6. Run only the cluster preflight job before any model download, inference, or
   training, followed by the explicitly labeled smoke chain.
7. Add small PowerShell helpers for sync, status, and result fetch only after
   the documented manual commands are confirmed.

## Official references

- [Northeastern: Connecting to Explorer](https://rc-docs.northeastern.edu/en/explorer-main/connectingtocluster/index.html)
- [Northeastern: Connecting on Windows](https://rc-docs.northeastern.edu/en/latest/connectingtocluster/windows.html)
- [Northeastern: Explorer cluster usage](https://rc-docs.northeastern.edu/en/explorer-main/best-practices/clusterusage.html)
- [Northeastern: H200 quick start](https://rc-docs.northeastern.edu/en/latest/gpus/quickstart-h200.html)
- [Northeastern Research Computing status](https://rc.northeastern.edu/status-updates/)
- [VS Code: Remote development using SSH](https://code.visualstudio.com/docs/remote/ssh)
- [VS Code: Remote development troubleshooting](https://code.visualstudio.com/docs/remote/troubleshooting)
- [VS Code: Profiles](https://code.visualstudio.com/docs/configure/profiles)
