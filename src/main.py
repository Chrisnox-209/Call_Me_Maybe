from parse import json_to_data, ParsingPompt, ParsngFunctions


def main() -> None:
    data_prompt = json_to_data('../data/input/function_calling_tests.json')
    data_function = json_to_data('../data/input/functions_definition.json')
    ParsingPompt.parse_prompts(data_prompt)
    ParsngFunctions.parse_functions(data_function)


if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     print("ok")
    main()
