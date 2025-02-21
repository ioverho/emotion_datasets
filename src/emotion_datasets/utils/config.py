import pathlib
import dataclasses
import typing


@dataclasses.dataclass
class FileSystemConfig:
    output_dir: str | pathlib.Path = "output"
    log_dir: str | pathlib.Path = "${file_system.output_dir}/logs"
    downloads_dir: str | pathlib.Path = "${file_system.output_dir}/downloads"
    data_dir: str | pathlib.Path = "${file_system.output_dir}/data"


@dataclasses.dataclass
class HuggingFaceConfig:
    max_shard_size: int | str = "500MB"
    num_shards: typing.Optional[int] = None
    num_proc: typing.Optional[int] = None
    storage_options: dict = dataclasses.field(default_factory=lambda: {})


@dataclasses.dataclass
class ConfigBase:
    defaults: typing.List[typing.Any] = dataclasses.field(
        default_factory=lambda: [
            "_self_",
        ]
    )

    file_system: FileSystemConfig = dataclasses.field(
        default_factory=lambda: FileSystemConfig
    )  # type: ignore

    huggingface: HuggingFaceConfig = dataclasses.field(
        default_factory=lambda: HuggingFaceConfig,
    )  # type: ignore

    overwrite: bool = False
    print_config: bool = False
    debug: bool = False

    hydra: typing.Any = dataclasses.field(
        default_factory=lambda: {
            "job": {
                "name": "process_all_datasets",
                "chdir": False,
            },
            "run": {
                "dir": "${file_system.log_dir}/${hydra.job.name}/${now:%y%m%d %H%M%S}"
            },
            "verbose": "${debug}",
        }
    )
