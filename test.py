# test.py
import json
import traceback
from main import PrudenceAPI

if __name__ == "__main__":
    try:
        api = PrudenceAPI()
        result = api.decide("CUST_HIGH", "P004")
        print("===== 决策结果 =====")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print("发生错误：")
        traceback.print_exc()