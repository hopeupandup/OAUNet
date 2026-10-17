"""Console and file logging."""

import os
import time
from datetime import datetime


class Logger:
    def __init__(self, log_dir='result/logs', log_name=None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if log_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_path = os.path.join(log_dir, f'train_{timestamp}.log')
        else:
            self.log_path = os.path.join(log_dir, f'{log_name}.log')

        self.colors = {
            'info': '\033[94m',
            'train': '\033[92m',
            'warning': '\033[93m',
            'error': '\033[91m',
            'reset': '\033[0m'
        }

    def _get_timestamp(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def log(self, message, level='info'):
        timestamp = self._get_timestamp()
        level = level.lower()

        log_line = f"[{timestamp}] [{level.upper()}] {message}"

        color = self.colors.get(level, self.colors['info'])
        print(f"{color}{log_line}{self.colors['reset']}")

        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"{log_line}\n")

    def info(self, message):
        self.log(message, level='info')

    def train(self, message):
        self.log(message, level='train')

    def warning(self, message):
        self.log(message, level='warning')

    def error(self, message):
        self.log(message, level='error')

    def log_config(self, config):
        self.info("=" * 60)
        self.info("训练配置")
        self.info("=" * 60)
        for section, params in config.items():
            self.info(f"[{section}]")
            if isinstance(params, dict):
                for key, value in params.items():
                    self.info(f"  {key}: {value}")
            else:
                self.info(f"  {params}")
        self.info("=" * 60)


def test_logger():
    print(f"\n{'='*60}")
    print(f"测试日志工具")
    print(f"{'='*60}")

    logger = Logger()
    logger.info("这是一条信息")
    logger.train("训练日志")
    logger.warning("警告信息")
    logger.error("错误信息")

    config = {
        'training': {'epochs': 20, 'batch_size': 2},
        'model': {'name': 'convgru', 'hidden_channels': 64}
    }
    logger.log_config(config)

    print(f"\n日志文件: {logger.log_path}")
    print(f"{'='*60}")
    print(f"✅ 日志工具测试通过!")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_logger()
