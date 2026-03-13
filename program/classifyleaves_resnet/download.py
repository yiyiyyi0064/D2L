#下载leaves数据
import kagglehub

kagglehub.login()

path = kagglehub.competition_download('classify-leaves')

print(f"Data downloaded to: {path}")