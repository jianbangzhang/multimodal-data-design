import json



def save_jsonl_online(data_list: list[dict], output_path: str):
    with open(output_path, "a", encoding="utf-8") as f:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")
    print("saved:", output_path)


def save_final_json(data_list: list[dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    print("saved:", output_path)