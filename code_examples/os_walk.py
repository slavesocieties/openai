import os

# if you're not seeing any output when you run this, check your working directory!

for root, folders, files in os.walk('sample_dir'):
    print(root)
    for folder in folders:
        print(os.path.join(root, folder))
    for file in files:
        print(os.path.join(root, file))