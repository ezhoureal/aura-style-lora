# CANN Export Setup

This guide installs Huawei CANN Kit's offline model-conversion tools and the
Kirin 9030 platform plugin locally. The repository uses these tools to convert
ONNX models to CANN offline models (`.omc`).

The commands below install the verified CANN Kit 6.0.1.0 release documented in
Huawei's [CANN Kit development preparation guide](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations).

## Requirements

- 64-bit Ubuntu or another compatible x86-64 Linux distribution
- `curl`, `unzip`, and `sha256sum`
- About 1.3 GB of free disk space for the archives and extracted tools
- The repository root as the current working directory

The downloaded and extracted files live under `.cannkit_downloads/` and
`.cannkit_tools/`. Both directories are ignored by Git.

## Download

```bash
mkdir -p .cannkit_downloads

curl --fail --location --continue-at - \
  --output .cannkit_downloads/DDK-tools-next-6.0.1.0.zip \
  'https://contentcenter-vali-drcn.dbankcdn.cn/pvt_2/DeveloperAlliance_package_901_9/4d/v3/ChvyTaVxR6CvAV1lVLB1-w/DDK-tools-next-6.0.1.0.zip?HW-CC-KV=V1&HW-CC-Date=20260123T095006Z&HW-CC-Expire=315360000&HW-CC-Sign=2CB6189092FA3B3FF22F538945D7DDB975C6DBCA41F753474B7B7A28E686AA6A'

curl --fail --location --continue-at - \
  --output .cannkit_downloads/kirin9030-plugin-next-6.0.1.0.zip \
  'https://contentcenter-vali-drcn.dbankcdn.cn/pvt_2/DeveloperAlliance_package_901_9/f0/v3/m4i1iJERRn6Ni6U-zH88Kw/kirin9030-plugin-next-6.0.1.0.zip?HW-CC-KV=V1&HW-CC-Date=20251222T032403Z&HW-CC-Expire=315360000&HW-CC-Sign=FC69A5B1E9B920477B07393B604A54A8C22F3A6F7F3E27EF2EF0B2026310A867'
```

Huawei may replace these artifacts in a future CANN Kit release. When updating
the version, use the archive URLs and checksums published in the preparation
guide rather than reusing the values below.

## Verify the archives

```bash
sha256sum .cannkit_downloads/*.zip
```

Expected output:

```text
1b2822fb9e5fe7443782915c6f34b4a2ce5c028207e7782514bd93970ff8e48a  .cannkit_downloads/DDK-tools-next-6.0.1.0.zip
3b32effc5af9804628cb9287e88cc28ed381877adb15dd85bf8d66e3be805251  .cannkit_downloads/kirin9030-plugin-next-6.0.1.0.zip
```

Do not install an archive if its checksum differs.

## Install

Extract the DDK first, then place the plugin under the DDK's `tools/platform`
directory:

```bash
mkdir -p .cannkit_tools/ddk
unzip -q .cannkit_downloads/DDK-tools-next-6.0.1.0.zip \
  -d .cannkit_tools/ddk

mkdir -p .cannkit_tools/ddk/tools/platform
unzip -q .cannkit_downloads/kirin9030-plugin-next-6.0.1.0.zip \
  -d .cannkit_tools/ddk/tools/platform

chmod u+x .cannkit_tools/ddk/tools/tools_omg/omg
```

The resulting files used by the export pipeline are:

```text
.cannkit_tools/ddk/tools/
├── platform/kirin9030/
├── tools_ascendc/
├── tools_dopt/
└── tools_omg/omg
```

Reinstalling the same version is safe after removing `.cannkit_tools/ddk`.
Keep the downloaded archives if you want to reinstall without downloading
them again.

## Verify the installation

Confirm that the wrapper can load the Kirin 9030 plugin:

```bash
.cannkit_tools/ddk/tools/tools_omg/omg \
  --help \
  --platform=kirin9030
```

The command should exit successfully and print the OMG usage information. An
error saying that `tools/platform/kirin9030` does not exist means the plugin was
extracted into the wrong directory.

To test an actual ONNX conversion:

```bash
source .cannkit_tools/ddk/tools/tools_ascendc/set_ascendc_env.sh

.cannkit_tools/ddk/tools/tools_omg/omg \
  --model=/path/to/model.onnx \
  --framework=5 \
  --platform=kirin9030 \
  --target=omc \
  --output=/tmp/cann-smoke-test

ls -lh /tmp/cann-smoke-test.omc
```

`--framework=5` selects ONNX. The OMG wrapper adds the `.omc` suffix to the
output name.

## Repository configuration

The default Hydra configuration in `configs/export/export.yaml` already uses
the local installation:

```yaml
export:
  quantization:
    dopt_path: .cannkit_tools/ddk/tools/tools_dopt/dopt_onnx_py3/dopt_so.py
  cann:
    platform: kirin9030
    omg_path: .cannkit_tools/ddk/tools/tools_omg/omg
    target: omc
```

Run the configured export pipeline with:

```bash
uv run python -m export
```

Select another model configuration through Hydra when needed. For example:

```bash
uv run python -m export models=s3od-dis
```

The pipeline sources `tools_ascendc/set_ascendc_env.sh` automatically before
invoking OMG, so no persistent shell environment changes are required.

## Troubleshooting

- `Permission denied` for `tools_omg/omg`: rerun the `chmod u+x` command.
- `platform/kirin9030 not exists`: extract the plugin directly into
  `.cannkit_tools/ddk/tools/platform`, not the DDK root.
- Checksum mismatch: delete the affected archive and download it again.
- Unsupported operators: inspect OMG's pre-check report and replace or
  decompose unsupported ONNX operators; installing the plugin does not make
  every ONNX operator NPU-compatible.
- Conversion succeeds with warnings: review the output model's CPU/NPU
  partitioning. A successful conversion does not guarantee that every operator
  runs on the NPU.
