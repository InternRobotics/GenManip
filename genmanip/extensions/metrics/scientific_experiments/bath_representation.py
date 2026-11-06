"""Read bath validity without advancing simulation or changing scene state."""


def bath_progress_valid(tube, namespace):
    mode = tube.GetAttribute("repair:bath_representation")
    validity_name = (
        "repair:bath_valid"
        if mode and mode.Get() == "static_visual"
        else f"{namespace}:pbd_valid"
    )
    valid = tube.GetAttribute(validity_name)
    error = tube.GetAttribute(f"{namespace}:runtime_error")
    return bool(valid and valid.Get() is True and not (error and error.Get()))
