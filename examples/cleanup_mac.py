import subprocess

commands = [
    "conda deactivate",
    "conda activate hdeo",
    "pip uninstall hyperbolic-tsne",    
    "rm -r ../hyperbolic_tsne.egg-info build",
    "rm ../hyperbolicTSNE/tsne_barnes_hut_hyperbolic.c* ../hyperbolicTSNE/tsne_barnes_hut.c* ../hyperbolicTSNE/tsne_utils.c*",
    "cd .. && python setup.py build_ext --inplace",
    "cd .. && pip install ."
]

for command in commands:
    # Run the terminal command
    try:
        print(command)
        full_command = "source /Users/chadepl/opt/miniconda3/bin/activate hdeo && " + command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Check the return code to see if the command was successful
        if result.returncode == 0:
            # Output of the command (stdout)
            print("Command output:")
            print(result.stdout)
        else:
            # Error output (stderr)
            print("Error occurred:")
            print(result.stderr)

    except Exception as e:
        print(f"An error occurred: {e}")