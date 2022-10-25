from dataclasses import dataclass

from omegaconf import OmegaConf


# 코드 외부에 있는 yaml 설정 파일에 포함된 변수의 정보를 명시하여, 해당 코드 내부에서 이를 인식할 수 있도록 기능
@dataclass
class Cfg:
    patch_size: int = 16
    channels: int = 3
    embed_dim: int = 128
    base_lr: float = 1e-4
    warmup_lr: float = 1e-7
    warmup_epochs: int = 5
    total_steps: int = 1000
    batch_size: int = 32
    epochs: int = 30
    patience: int = 5
    clip_grad: float = 3.0
    temperature: float = 0.1
    log_interval: int = 50


# OmegaConf 자료형으로 개별 멤버 값에 직접적으로 접근이 불가능
# -> 명확하게 반환형이 Cfg 클래스는 아니지만, duck typing을 통해 Cfg 클래스 객체인 것처럼 작동함..
def load_config(path: str) -> Cfg:
    schema = OmegaConf.structured(Cfg())

    # duck typing
    config: Cfg = OmegaConf.merge(schema, OmegaConf.load(path))

    return config
