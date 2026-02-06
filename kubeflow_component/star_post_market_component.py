from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model

# Insert your dockerhub image below (e.g. "docker.io/<username>/<image_name>:<tag>")
DOCKER_IMAGE = "<docker_image>"


@dsl.component(base_image="python:3.14-slim")
def download_repo(
        github_repo_url: str,
        project_files: Output[Model],
        data: Output[Dataset],
        branch: str = "main",
) -> None:
    """
    Download specific scripts and data from a GitHub repository.

    :param github_repo_url: URL of the GitHub repository to clone.
    :param project_files: Output path for project scripts.
    :param data: Output path for data folder.
    :param branch: Branch name to pull from (defaults to 'main').
    """
    import shutil
    from pathlib import Path
    import subprocess

    repo_dir = Path("/tmp/repo")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    print("Installing git...")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "git"], check=True)

    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            branch,
            "--single-branch",
            github_repo_url,
            str(repo_dir),
        ],
        check=True,
    )
    print(f"Cloned repo {github_repo_url} (branch: {branch}).")

    proj_path = Path(project_files.path)
    proj_path.mkdir(parents=True, exist_ok=True)
    src_folder = repo_dir / "src"

    if src_folder.exists():
        for item in src_folder.iterdir():
            if item.is_file():
                shutil.copy2(item, proj_path / item.name)
                print(f"Copied src/{item.name}")
            elif item.is_dir():
                shutil.copytree(item, proj_path / item.name, dirs_exist_ok=True)
                print(f"Copied src/{item.name}/ directory")
    else:
        print("Warning: src/ folder not found in repo")

    required_files = [
        "adversarial_evaluation.py",
        "expert_knowledge.py",
        "statistical_analysis.py",
        "STAR_model.py",
        "utils/generic_utils.py",
        "utils/data_helpers.py",
        "utils/time_conversion.py",
    ]

    missing_files = []
    for file_path in required_files:
        full_path = proj_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
            print(f"ERROR: Required file missing: {file_path}")
        else:
            print(f"✓ Verified: {file_path}")

    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")

    data_path = Path(data.path)
    data_path.mkdir(parents=True, exist_ok=True)
    src_data_path = repo_dir / "data"

    if src_data_path.exists():
        for item in src_data_path.iterdir():
            if item.is_file():
                shutil.copy2(item, data_path / item.name)
                print(f"Copied data/{item.name}")
            elif item.is_dir():
                shutil.copytree(item, data_path / item.name, dirs_exist_ok=True)
                print(f"Copied data/{item.name}/ directory")
    else:
        print("Warning: data folder not found in repo")


@dsl.component(base_image="python:3.14-slim")
def expert_knowledge_evaluation(
        project_files: Input[Model],
        data: Input[Dataset],
        expert_knowledge_results: Output[Dataset],
) -> None:
    """
    Run expert knowledge evaluation on synthetic data.

    :param project_files: Input containing project scripts from repository.
    :param data: Input dataset containing synthetic data files.
    :param expert_knowledge_results: Output path for expert knowledge results.
    """
    from pathlib import Path
    import subprocess

    proj_path = Path(project_files.path)
    data_path = Path(data.path)
    results_path = Path(expert_knowledge_results.path)
    results_path.mkdir(parents=True, exist_ok=True)

    script = proj_path / "expert_knowledge.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Expert Knowledge evaluation script not found at {script}"
        )

    cmd = [
        "python",
        str(script),
        "--synth_dir",
        str(data_path / "synthetic_data"),
        "--output",
        str(results_path / "expert_knowledge_results.json"),
    ]
    subprocess.run(cmd, check=True)

    print(f"Expert Knowledge evaluation finished. Results saved to {results_path}")


@dsl.component(base_image="python:3.14-slim")
def statistical_analysis(
        project_files: Input[Model],
        data: Input[Dataset],
        statistical_results: Output[Dataset],
) -> None:
    """
    Run comprehensive statistical analysis on synthetic data for quality assessment.

    :param project_files: Input containing project scripts from repository.
    :param data: Input dataset containing synthetic data files.
    :param statistical_results: Output path for statistical analysis results.
    """
    from pathlib import Path
    import subprocess

    proj_path = Path(project_files.path)
    data_path = Path(data.path)
    results_path = Path(statistical_results.path)
    results_path.mkdir(parents=True, exist_ok=True)

    script = proj_path / "statistical_analysis.py"
    if not script.exists():
        raise FileNotFoundError(f"Statistical analysis script not found at {script}")

    cmd = [
        "python",
        str(script),
        "--synth_dir",
        str(data_path / "synthetic_data"),
        "--output",
        str(results_path / "statistical_analysis_results.json"),
    ]
    subprocess.run(cmd, check=True)

    print(f"Statistical analysis finished. Results saved to {results_path}")


@dsl.container_component
def adversarial_evaluation(
        project_files: Input[Model],
        data: Input[Dataset],
        adversarial_evaluation_results: Output[Dataset],
):
    """
    Run adversarial evaluation comparing synthetic vs real-world data performance.

    :param project_files: Input containing project scripts from repository.
    :param data: Input dataset containing synthetic and real-world data files.
    :param adversarial_evaluation_results: Output path for adversarial evaluation results.
    """
    command_str = f"""
        set -e
        apt-get update
        apt-get install -y python3 python3-dev wget curl
        curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --break-system-packages
        python3 -m pip install --break-system-packages pandas==3.0.0 scikit-learn==1.8.0
        cd {project_files.path}

        python3 adversarial_evaluation.py \
            --synth_dir {data.path}/synthetic_data \
            --rwd_dir {data.path}/rwd_data \
            --output {adversarial_evaluation_results.path}/adversarial_evaluation_results.json \
            --docker_image {DOCKER_IMAGE} \
            --in_docker True
        ls -la {adversarial_evaluation_results.path}
    """

    return dsl.ContainerSpec(
        image=DOCKER_IMAGE,
        command=["sh", "-c"],
        args=[command_str]
    )


@dsl.pipeline(
    name="STAR Post-Market Evaluation Pipeline",
    description="Runs expert knowledge, statistical analysis, and adversarial evaluation checks.",
)
def star_post_market_pipeline(
        github_repo_url: str,
        branch: str = "main",
):
    """
    STAR Post-Market Evaluation Pipeline for synthetic data validation.

    :param github_repo_url: URL of the GitHub repository containing evaluation scripts.
    :param branch: Git branch to pull from repository (defaults to 'main').
    """
    repo_task = download_repo(github_repo_url=github_repo_url, branch=branch)
    repo_task.set_caching_options(False)
    repo_task.set_cpu_request("1000m")
    repo_task.set_cpu_limit("2000m")
    repo_task.set_memory_request("2Gi")
    repo_task.set_memory_limit("4Gi")

    expert_knowledge_task = expert_knowledge_evaluation(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
    )
    expert_knowledge_task.after(repo_task)
    expert_knowledge_task.set_caching_options(False)
    expert_knowledge_task.set_cpu_request("1000m")
    expert_knowledge_task.set_cpu_limit("2000m")
    expert_knowledge_task.set_memory_request("2Gi")
    expert_knowledge_task.set_memory_limit("4Gi")

    statistical_task = statistical_analysis(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
    )
    statistical_task.after(repo_task)
    statistical_task.set_caching_options(False)
    statistical_task.set_cpu_request("1000m")
    statistical_task.set_cpu_limit("2000m")
    statistical_task.set_memory_request("2Gi")
    statistical_task.set_memory_limit("4Gi")

    adversarial_task = adversarial_evaluation(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
    )
    adversarial_task.after(repo_task)
    adversarial_task.set_caching_options(False)
    adversarial_task.set_cpu_request("4000m")
    adversarial_task.set_cpu_limit("6000m")
    adversarial_task.set_memory_request("8Gi")
    adversarial_task.set_memory_limit("12Gi")


if __name__ == "__main__":
    kfp_compiler = compiler.Compiler()
    kfp_compiler.compile(
        pipeline_func=star_post_market_pipeline,
        package_path="star_post_market_pipeline.yaml",
    )
