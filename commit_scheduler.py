import os
import sys
import json
import subprocess
import argparse

STATE_FILE = "commit_state.json"

# Define the 6 groups of files to push daily (~15% of files per day)
FILE_GROUPS = [
    ["requirements.txt", ".env"],
    ["config.py", "database.py"],
    ["data_simulator.py", "features.py"],
    ["model.py", "risk.py"],
    ["backtester.py", "dashboard.py"],
    ["main.py", "tests/test_engine.py", "commit_scheduler.py"]
]

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"current_day": 0, "remote_url": ""}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error executing command '{cmd}':\n{result.stderr.strip()}")
        return False, result.stderr.strip()
    return True, result.stdout.strip()

def initialize_git(remote_url=None):
    print("Checking git repository initialization...")
    if not os.path.exists(".git"):
        success, _ = run_command("git init")
        if not success:
            return False
        run_command("git branch -M main")
        print("Initialized new git repository with branch 'main'.")
    else:
        print("Git repository already exists.")

    if remote_url:
        # Check if remote already exists
        _, remotes = run_command("git remote")
        if "origin" in remotes:
            run_command("git remote remove origin")
        success, _ = run_command(f"git remote add origin {remote_url}")
        if success:
            print(f"Set remote origin to: {remote_url}")
            return True
        return False
    return True

def push_daily_increment():
    state = load_state()
    day = state["current_day"]
    remote_url = state["remote_url"]

    # 1. Check if git is initialized
    initialize_git(remote_url if remote_url else None)

    # 2. Check if we have completed all days
    if day >= len(FILE_GROUPS):
        print(f"All {len(FILE_GROUPS)} days of file uploads have been processed! Git history is fully aligned.")
        return

    files_to_add = FILE_GROUPS[day]
    print(f"\n--- RUNNING RELEASE: DAY {day + 1} of {len(FILE_GROUPS)} ---")
    print(f"Files to commit: {', '.join(files_to_add)}")

    # Verify if files exist locally before committing
    missing_files = [f for f in files_to_add if not os.path.exists(f)]
    if missing_files:
        print(f"Warning: The following files are missing and will be skipped: {', '.join(missing_files)}")
        files_to_add = [f for f in files_to_add if os.path.exists(f)]

    if not files_to_add:
        print("No files available to commit for today.")
        return

    # 3. Stage the files
    for file in files_to_add:
        run_command(f"git add '{file}'")
    
    # Also save state changes to make sure state is tracked in the repository
    run_command(f"git add '{STATE_FILE}'")

    # 4. Commit files
    commit_msg = f"[Day {day + 1}/6 Release] Adding {', '.join([os.path.basename(f) for f in files_to_add])}"
    success, stdout = run_command(f'git commit -m "{commit_msg}"')
    if not success:
        if "nothing to commit" in stdout or "clean" in stdout:
            print("No changes detected or files already committed.")
        else:
            print("Commit failed. Check file modifications.")
            return

    # 5. Push to remote if configured
    if remote_url:
        print(f"Pushing to remote repository: {remote_url}...")
        push_success, push_out = run_command("git push -u origin main")
        if push_success:
            print(f"Successfully pushed Day {day + 1} release to remote main branch!")
        else:
            print("Push failed! Please ensure the remote GitHub repository is empty and your SSH/HTTPS credentials are correct.")
            print(f"Detailed error: {push_out}")
            print("State has NOT been incremented. Fix remote settings and re-run.")
            return
    else:
        print("Notice: No remote URL configured yet. Files committed locally.")
        print("Configure a remote repository to push files online using:")
        print("  python commit_scheduler.py --set-remote <your-github-repo-url>")

    # 6. Update state
    state["current_day"] = day + 1
    save_state(state)
    print(f"Day {day + 1} release state saved. Next run will be Day {day + 2}.")

def main():
    parser = argparse.ArgumentParser(description="Automate 15% daily incremental pushes to GitHub.")
    parser.add_argument("--set-remote", type=str, help="Configure or update the remote GitHub repository URL")
    parser.add_argument("--run", action="store_true", help="Execute the scheduled commit and push for today")
    parser.add_argument("--status", action="store_true", help="Display the current scheduling release status")
    
    args = parser.parse_args()

    state = load_state()

    if args.set_remote:
        # Validate URL
        url = args.set_remote.strip()
        state["remote_url"] = url
        save_state(state)
        # Initialize git config remote
        initialize_git(url)
        print(f"Remote URL configured: {url}")
        
    elif args.status:
        day = state["current_day"]
        remote = state["remote_url"] or "Not configured"
        print("=" * 45)
        print("      COMMIT & PUSH RELEASE SCHEDULER")
        print("=" * 45)
        print(f"Current Day: {day} of {len(FILE_GROUPS)}")
        print(f"Remote URL : {remote}")
        if day < len(FILE_GROUPS):
            print(f"Next Files : {', '.join(FILE_GROUPS[day])}")
            percentage = (day / len(FILE_GROUPS)) * 100
            print(f"Completion : {percentage:.1f}%")
        else:
            print("Completion : 100.0% (All days complete)")
        print("=" * 45)
        
    elif args.run:
        push_daily_increment()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
