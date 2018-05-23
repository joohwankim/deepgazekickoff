# Step 1. Capture
#   a. Synthetic data. Animate desired blender scenes.
#       Open script_home.blend, open IterateOnModels.py in [git-repo]/capture/synthetic/blender/, and run.
#   b. Real data. Capture real eye data.
#       Generate .csv files for the capture plan and record the video with the subject.

# Step 2. Footage
#   a. Synthetic data. Render the animated scene on SaturnV nodes by running SubmitBlenderJobs.py in [git-repo]/footage/synthetic/blender
#   b. Real data. Convert video files to zip (this step may go into the capture app)

# Step 3. Datasets
#   Run data_prep.py in [git-repo]/datasets

# Step 4. Training
#   Run runTraining.py for training locally or submitJob.py for training on SaturnV nodes. Run scratch.py for not recording the outputs.
