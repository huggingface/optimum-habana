import argparse
import json
import os
import sys

import numpy as np


def find_measurement_path(measurement, measurements_dir_path, scales, group_size):
    measurment_card = measurement + "_" + str(group_size)
    for measurment_file in os.listdir(measurements_dir_path):
        filename = os.fsdecode(measurment_file)
        if not filename.endswith(".json") or "_mod_list" in filename or measurment_card not in filename:
            continue
        if scales:
            if "MAXABS" in filename:
                return os.path.join(measurements_dir_path, measurment_file)
        else:
            if "MAXABS" not in filename:
                return os.path.join(measurements_dir_path, measurment_file)


def unify_measurements(measurement_group, measurements_dir_path, output_path, scales=False):
    measurements_paths = []
    group_name = ""

    # save all the jsons paths in the given measurement group
    for measurement in measurement_group:
        measurement_path = find_measurement_path(measurement, measurements_dir_path, scales, len(measurement_group))
        measurements_paths.append(measurement_path)
        group_name += measurement

    # save all the jsons content in the given measurement group
    measurements_jsons = []
    for measurement_path in measurements_paths:
        with open(measurement_path, "r") as f:
            js = json.load(f)
            measurements_jsons.append(js["Nodes"])

    # create a name for the unified json that will be created for this measurement group
    unified_json_name = (
        find_measurement_path(measurement_group[0], measurements_dir_path, scales, len(measurement_group))
        .split("/")[-1]
        .replace("_" + measurement_group[0] + "_" + str(len(measurement_group)), "")
    )
    unified_json_path = os.path.join(output_path, unified_json_name)

    # open a unified json file
    with open(measurements_paths[0], "r") as origin, open(unified_json_path, "w") as copy:
        copy.write(origin.read())
    with open(unified_json_path, "r") as json_file:
        unified_json = json.load(json_file)

    # iterate all unified json nodes
    for node_name, node_values in unified_json["Nodes"].items():
        max_inputs = node_values["inputs"]
        max_outputs = None
        if node_values.get("outputs") is not None:
            max_outputs = node_values["outputs"]
        max_weight = None
        if node_values.get("params") is not None and node_values["params"].get("weight") is not None:
            max_weight = node_values["params"]["weight"]

        # iterate over all the measurment group and take the maximum for each tensor and its channel
        if scales:
            for measurement_json in measurements_jsons:
                max_inputs[0] = max(measurement_json[node_name]["inputs"][0], max_inputs[0])
                if max_outputs is not None:
                    max_outputs = max(measurement_json[node_name]["outputs"], max_outputs)
                if max_weight is not None:
                    max_weight = max(measurement_json[node_name]["params"]["weight"], max_weight)
        else:
            for measurement_json in measurements_jsons:
                for i in range(0, len(max_inputs)):
                    for j in range(0, len(max_inputs[i])):
                        max_inputs[i][j][0] = max(measurement_json[node_name]["inputs"][i][j][0], max_inputs[i][j][0])
                if max_outputs is not None:
                    for i in range(0, len(max_outputs)):
                        max_outputs[i][0] = max(measurement_json[node_name]["outputs"][i][0], max_outputs[i][0])
                if max_weight is not None:
                    for i in range(0, len(max_weight)):
                        max_weight[i][0] = max(measurement_json[node_name]["params"]["weight"][i][0], max_weight[i][0])

        # update the maximum in the unified json
        if scales:
            unified_json["Nodes"][node_name]["inputs"][0] = max_inputs[0]
            if max_outputs is not None:
                unified_json["Nodes"][node_name]["outputs"] = max_outputs
            if max_weight is not None:
                unified_json["Nodes"][node_name]["params"]["weight"] = max_weight
        else:
            for i in range(0, len(max_inputs)):
                for j in range(0, len(max_inputs[i])):
                    unified_json["Nodes"][node_name]["inputs"][i][j][0] = max_inputs[i][j][0]
            if max_outputs is not None:
                for i in range(0, len(max_outputs)):
                    unified_json["Nodes"][node_name]["outputs"][i][0] = max_outputs[i][0]
            if max_weight is not None:
                for i in range(0, len(max_weight)):
                    unified_json["Nodes"][node_name]["params"]["weight"][i][0] = max_weight[i][0]
    global_rank = None
    local_rank = None
    mode = ""
    layers = {}
    with open(unified_json_path, "w") as json_file:
        json.dump(unified_json, json_file)
    mode = unified_json["Mode"]
    nodes = unified_json["Nodes"]

    # create unified npz file from the unified json
    unified_npz_path = os.path.join(output_path, unified_json_name.replace(".json", ".npz"))
    for layer, dlayer in nodes.items():
        layers[layer] = {}
        layers[layer]["inputs"] = [np.array(x) for x in dlayer["inputs"]]
        if dlayer.get("outputs") is not None:
            layers[layer]["outputs"] = np.array(dlayer["outputs"])
        if dlayer.get("params") is not None and dlayer["params"].get("weight") is not None:
            layers[layer]["params"] = {}
            layers[layer]["params"]["weight"] = np.array(dlayer["params"]["weight"])
    df = {"GlobalRank": global_rank, "LocalRank": local_rank, "Mode": mode, "Nodes": layers}
    with open(unified_npz_path, "w"):
        np.savez(unified_npz_path, df)


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Run the measurements parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m", "--measurements", type=str, help="path to the directory of the measurements that will be unified"
    )
    parser.add_argument(
        "-g",
        "--groups",
        type=list,
        nargs="+",
        help="the groups of cards that are going to be unified- e.g. 01 23 45 67",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=os.getcwd(),
        help="path to the directory where the unified measurements will be written",
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    output_path = args.out
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    measurements_path = args.measurements
    groups = args.groups

    num_jsons = 0
    for path in os.listdir(measurements_path):
        if path.endswith(".json"):
            num_jsons += 1
    assert os.path.isdir(measurements_path) and (num_jsons % len(groups)) == 0

    for group in groups:
        unify_measurements(group, measurements_path, output_path, scales=False)
        unify_measurements(group, measurements_path, output_path, scales=True)

    print("finished measurement unifier script")


if __name__ == "__main__":
    main(sys.argv[1:])
