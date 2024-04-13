# Video Diffusion

## How to run this on EC2

1. Create an EC2 instance with GPU

2. Use PyCharm's deployment and send files via SFTP

    - ssh into the EC2 instance
    - Now you should see this folder
    - `cd` into this folder
    - You should see a setup script named `setup.sh`

3. Run `bash setup.sh`

    - This will download data, install dependencies, etc.

4. Activate venv:

   `. .venv/bin/activate`

5. Now you should be able to run the Python code

   `python3 -m run.train`

## How to use rye

- Sync: `rye sync`
- Add dependency: `rye add numpy`
- Remove dependency: `rye remove numpy`
- Activate the virtualenv: `. .venv/bin/activate`
- Deactivate: `deactivate`

## How to use wandb

- In training, we use wandb for logging
- To use it, you need to register an account on wandb's website
- You can get its API key and input the key when the code prompts you