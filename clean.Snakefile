SEQUENCES = [
    "sequence-1",
    "sequence-2",
    "sequence-3",
    "sequence-4",
    "sequence-5",
    "sequence-6",
    "sequence-7",
    "sequence-8",
    "sequence-9",
    "sequence-10",
    "sequence-11",
    "sequence-12",
    "sequence-13",
]

CLEAN_FLAGS = {
    "sequence-11": "--skip 25",
}

HF_REPO = "ucu-autonomous-ugv/ucu-slam-dataset-v1"


rule all:
    input:
        expand("data/hf-logs/{seq}.txt", seq=SEQUENCES),
        expand("data/previews/{seq}-preview.png", seq=SEQUENCES),
        expand("data/previews/{seq}-map.png", seq=SEQUENCES),


rule clean_sequence:
    input:
        raw_dir="data/raw/{seq}",
        calibration="calibration/calibration.yaml",
        face_model="models/ego_blur_face_gen2.jit",
        license_plate_model="models/ego_blur_lp_gen2.jit",
    output:
        clean_dir=temp(directory("data/clean/{seq}")),
    resources:
        disk_mb=25_000,
    params:
        flags=lambda wildcards: CLEAN_FLAGS.get(wildcards.seq, ""),
    shell:
        """
        echo 'Cleaning {wildcards.seq} with flags: {params.flags}'
        python3 scripts/clean.py {input.raw_dir} {output.clean_dir} {params.flags} --force --face-model-path {input.face_model} --license-plate-model-path {input.license_plate_model} --calibration-file {input.calibration} --device "${{DEVICE:-cpu}}"
        """


rule generate_preview:
    input:
        clean_dir="data/clean/{seq}",
    output:
        preview="data/previews/{seq}-preview.png",
        map="data/previews/{seq}-map.png",
    resources:
        disk_mb=25_000,
    shell:
        """
        python3 scripts/generate_preview.py {input.clean_dir} --output-preview-path {output.preview} --output-trajectory-path {output.map}
        """


rule push_hf:
    input:
        clean_dir="data/clean/{seq}",
    output:
        hf_log="data/hf-logs/{seq}.txt",
    resources:
        disk_mb=25_000,
    params:
        hf_repo=HF_REPO,
    shell:
        """
        HF_HUB_DISABLE_XET=1 hf upload {params.hf_repo} {input.clean_dir} /{wildcards.seq} --repo-type=dataset

        echo "Successfully processed and uploaded {wildcards.seq} to {params.hf_repo}" > {output.hf_log}
        """
