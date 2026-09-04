SEQUENCES = ["sequence-a1", "sequence-a2", "sequence-a3"]
SEQUENCES_PREVIEW = ["sequence-a1", "sequence-a3"]

CLEAN_FLAGS = {
    "sequence-a2": "--skip 100 --max-seconds 2110",
}
PREVIEW_FLAGS = {
    "sequence-a1": "--preview-at 0.28",
    "sequence-a3": "--preview-at 0.28",
}

HF_REPO = "ucu-autonomous-ugv/ucu-slam-dataset-v1"

rule all:
    input:
        expand("data/hf-logs/{seq}.txt", seq=SEQUENCES),
        expand("data/previews/{seq}-preview.png", seq=SEQUENCES_PREVIEW),


rule clean_sequence:
    input:
        raw_dir = "data/raw/{seq}"
    output:
        clean_dir = directory("data/clean/{seq}")
    resources:
        disk_mb = 25_000
    params:
        flags = lambda wildcards: CLEAN_FLAGS.get(wildcards.seq, "")
    shell:
        """
        echo 'Cleaning {wildcards.seq} with flags: {params.flags}'
        python3 scripts/clean.py {input.raw_dir} {output.clean_dir} {params.flags} --force
        """


rule convert_ros1:
    input:
        clean_dir = "data/clean/{seq}"
    output:
        ros1_bag = "data/ros1/{seq}.bag"
    resources:
        disk_mb = 25_000
    shell:
        "rosbags-convert --src {input.clean_dir} --dst {output.ros1_bag}"


rule generate_preview:
    input:
        clean_dir = "data/clean/{seq}"
    output:
        preview = "data/previews/{seq}-preview.png"
    resources:
        disk_mb = 25_000
    params:
        flags = lambda wildcards: PREVIEW_FLAGS.get(wildcards.seq, "")
    shell:
        """
        echo 'Generating preview for {wildcards.seq} with flags: {params.flags}'
        python3 scripts/generate_preview.py {input.clean_dir} --output-preview-path {output.preview} {params.flags}
        """


rule push_hf:
    input:
        clean_dir = "data/clean/{seq}",
        ros1_bag = "data/ros1/{seq}.bag",
    output:
        hf_log = "data/hf-logs/{seq}.txt"
    resources:
        disk_mb = 25_000
    params:
        hf_repo = HF_REPO
    shell:
        """
        HF_HUB_DISABLE_XET=1 hf upload {params.hf_repo} {input.clean_dir} /{wildcards.seq} --repo-type=dataset
        HF_HUB_DISABLE_XET=1 hf upload {params.hf_repo} {input.ros1_bag} /{wildcards.seq}.bag --repo-type=dataset

        echo "Successfully processed and uploaded {wildcards.seq} to {params.hf_repo}" > {output.hf_log}
        """
