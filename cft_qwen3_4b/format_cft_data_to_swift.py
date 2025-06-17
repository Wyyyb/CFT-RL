import json
import os


def format_single(item):
    instruction = item["instruction"]
    input_str = item["input"]
    output_str = item["output"]
    result = {"messages":
        [
            {"role": "user", "content": instruction + "\n\n" + input_str},
            {"role": "assistant", "content": output_str}
        ]
    }
    return result


def main():
    with open("../local_data/webinstruct_cft_50k_0129.json", "r") as fi:
        data = json.load(fi)
    res = []
    for each in data:
        res.append(format_single(each))
    with open("../local_data/webinstruct_cft_50k_swift_0617.json", "w") as fo:
        fo.write(json.dumps(res, indent=4))


if __name__ == "__main__":
    main()






























