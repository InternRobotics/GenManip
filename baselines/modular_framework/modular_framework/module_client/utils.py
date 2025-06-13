def process_check_finished(result):
    while isinstance(result, dict):
        if "finished" in result:
            result = result["finished"]
        else:
            for res in result:
                result = result[res]
                break
    return result