import os
import tempfile
import shutil
import importlib
import json
import yaml


class Config(dict):
    @staticmethod
    def from_file(filename):
        """
        파일 확장자에 따라 설정 파일을 읽고 Config 객체로 변환
        """
        # 파일 확장자에 따른 처리
        if filename.endswith(".py"):
            return Config.__file2dict_py(filename)
        elif filename.endswith(".json"):
            return Config.__file2dict_json(filename)
        elif filename.endswith((".yml", ".yaml")):
            return Config.__file2dict_yaml(filename)
        else:
            raise ValueError(f"Unsupported file format: {filename}")

    @staticmethod
    def __file2dict_py(filename):
        """
        Python 파일을 읽어 딕셔너리로 변환
        """
        temp_module_name = None
        try:
            # 임시 디렉터리에 Python 파일 복사
            temp_dir = tempfile.mkdtemp()
            shutil.copy(filename, temp_dir)
            temp_module_name = os.path.splitext(os.path.basename(filename))[0]
            temp_module_path = os.path.join(temp_dir, temp_module_name)

            # Python 파일을 모듈로 임포트
            spec = importlib.util.spec_from_file_location(
                temp_module_name, temp_module_path + ".py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 모듈에서 변수 추출
            cfg_dict = {
                name: value
                for name, value in vars(module).items()
                if not name.startswith("__")
            }
            return Config(cfg_dict)

        finally:
            # 임시 디렉터리 정리
            if temp_module_name:
                shutil.rmtree(temp_dir)

    @staticmethod
    def __file2dict_json(filename):
        """
        JSON 파일을 읽어 딕셔너리로 변환
        """
        with open(filename, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        return Config(cfg_dict)

    @staticmethod
    def __file2dict_yaml(filename):
        """
        YAML 파일을 읽어 딕셔너리로 변환
        """
        with open(filename, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        return Config(cfg_dict)
