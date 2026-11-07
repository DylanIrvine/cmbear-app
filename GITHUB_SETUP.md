# Getting this folder onto GitHub and Streamlit Community Cloud

Target: https://github.com/DylanIrvine/cmbear-app (public), deployed at share.streamlit.io.

## A. Publish with GitHub Desktop (recommended)

1. Open GitHub Desktop. Check you are signed in: File > Options > Accounts should show DylanIrvine.
2. File > Add local repository. Browse to `I:\Projects\Apps\CMBEAR\cmbear-app` and choose Add.
   Desktop will say the folder is not a Git repository and offer "create a repository here instead".
   Click that link.
3. In the Create a repository dialog:
   - Name: cmbear-app (leave as is)
   - Description: CMBEAR: chloride mass balance recharge estimates for Australian bores
   - Initialize this repository with a README: leave unticked (one exists)
   - Git ignore: None (one exists)
   - License: None (one exists)
   Click Create repository.
4. The left panel now lists about 20 changed files. In the Summary box bottom-left type
   `Initial CMBEAR Streamlit app`, then click Commit to main.
5. Click Publish repository (top bar). In the dialog:
   - Name: cmbear-app
   - Untick "Keep this code private"  (Streamlit Community Cloud needs a public repo)
   - Organization: None
   Click Publish repository.
6. Check it worked: Repository > View on GitHub opens https://github.com/DylanIrvine/cmbear-app
   showing the files and the README rendered below them.

## B. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and click Sign in with GitHub (or Continue with GitHub).
   First time only: GitHub asks you to authorise Streamlit; approve it.
2. Click Create app (top right), then "Deploy a public app from GitHub".
3. Fill in:
   - Repository: DylanIrvine/cmbear-app
   - Branch: main
   - Main file path: app.py
   - App URL: choose a subdomain, e.g. cmbear  (gives https://cmbear.streamlit.app)
   Advanced settings can be left alone (Python 3.12 default is fine; no secrets needed).
4. Click Deploy. The log shows pip installing requirements.txt; the first build takes 2 to 4 minutes.
   When it finishes the app opens. Upload examples/example_input.csv to test it.

## C. Making changes later

1. Edit files in the folder as usual.
2. In GitHub Desktop the changed files appear on the left. Write a one-line summary, Commit to main,
   then click Push origin (top bar).
3. Streamlit Cloud rebuilds automatically within a minute or two. If it looks stuck, open the app,
   click Manage app (bottom right) and choose Reboot.

## D. If something goes wrong

- Publish fails with an authentication error: File > Options > Accounts, sign out and back in.
- Streamlit build fails: open Manage app > logs. Almost always a missing package in requirements.txt.
- App shows "Loading national grids..." for a long time on first visit: normal on a cold start
  (the 13.6 MB archive is read once and cached); subsequent visits are instant.
- Streamlit Cloud puts apps to sleep after a period without visitors; the first visitor afterwards
  waits about 30 seconds while it wakes. Nothing is lost.

## Command-line alternative (Git for Windows)

    cd I:\Projects\Apps\CMBEAR\cmbear-app
    git init -b main
    git add .
    git commit -m "Initial CMBEAR Streamlit app"
    git remote add origin https://github.com/DylanIrvine/cmbear-app.git   # after creating an empty repo at github.com/new
    git push -u origin main

Notes: data/grids.npz (13.6 MB) is well under GitHub's 100 MB per-file limit, so no Git LFS is needed.
The Lee2024 folder is not part of the repository; only the derived grids.npz is.
