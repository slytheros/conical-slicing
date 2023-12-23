import argparse
import os
import pathlib
import re
import subprocess
import vtkplotlib
import tqdm
import numpy
import logging

from collections import OrderedDict
from stl import mesh

logging.basicConfig(filename='conical_slice.log')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def load_mesh(filename: str) -> mesh.Mesh:
    """
    Load a mesh from file and return it as a mesh.Mesh.

    :param filename: path to the STL file
    :return: a mesh.Mesh
    """
    logger.info(f"Loading mesh from {filename}")
    np_mesh = mesh.Mesh.from_file(filename)
    return np_mesh


def subdivide_triangle(triangle_points: numpy.ndarray) -> numpy.ndarray:
    """
    Subdivide a triangle in 3D space into 4 sub-triangles using the mid-points of all edges.

    :param triangle_points:
    :return: a numpy.ndarray containing the subdivided triangles
    """
    triangles = numpy.zeros((4, 3, 3))
    point1 = triangle_points[0]
    point2 = triangle_points[1]
    point3 = triangle_points[2]
    midpoint12 = (point1 + point2) / 2
    midpoint23 = (point2 + point3) / 2
    midpoint31 = (point3 + point1) / 2

    triangles[0] = numpy.array([point1, midpoint12, midpoint31])
    triangles[1] = numpy.array([point2, midpoint23, midpoint12])
    triangles[2] = numpy.array([point3, midpoint31, midpoint23])
    triangles[3] = numpy.array([midpoint12, midpoint23, midpoint31])
    return triangles


def subdivide(my_mesh: mesh.Mesh) -> mesh.Mesh:
    """
    Increases the polygon count of a mesh by subdividing each triangle into 4 sub-triangles.
    Maintains geometry (no interpolation)

    TODO: This is called iteratively and its execution time is increased exponentially with each iteration.
    TODO: Might be worth re-implementing to do selective subdivision for triangles with edges under a threshold

    :param my_mesh: the mesh to subdivide
    :return: the subdivided mesh
    """
    logger.info(f"Subdividing mesh")
    divided_vectors = numpy.zeros(shape=(my_mesh.vectors.shape[0] * 4, 3, 3))
    for ix, vector in tqdm.tqdm(enumerate(my_mesh.vectors), desc="Subdividing mesh"):
        divided_vectors[ix * 4:ix * 4 + 4] = subdivide_triangle(vector)

    # my_mesh['vectors'] = divided_vectors
    # my_mesh
    my_mesh_transformed = numpy.zeros(divided_vectors.shape[0], dtype=mesh.Mesh.dtype)
    my_mesh_transformed['vectors'] = divided_vectors
    my_mesh_transformed = mesh.Mesh(my_mesh_transformed)

    # mesh.Mesh(my_mesh.vectors)
    return my_mesh_transformed


def reset_origin(np_mesh: mesh.Mesh) -> mesh.Mesh:
    """
    Resets the origin of the mesh. The origin is set to the center of the object.
    :param np_mesh:
    :return:
    """
    x_delta = - np_mesh.x.min() - (np_mesh.x.max() - np_mesh.x.min()) / 2
    y_delta = - np_mesh.y.min() - (np_mesh.y.max() - np_mesh.y.min()) / 2

    logger.info(f"Resetting origin of mesh using offsets {x_delta}, {y_delta}")

    np_mesh.translate((x_delta,
                       y_delta,
                       -np_mesh.z.min()))

    return np_mesh


def stl_transform(np_mesh: mesh.Mesh, angle_degrees: int | float, cone_origin: numpy.array) -> mesh.Mesh:
    """
    Transforms the mesh by offsetting the z axis using by a cone transformation.
    :param np_mesh:
    :param angle_degrees:
    :param cone_origin: the position of the cone
    :return:
    """
    np_mesh.z += (np_mesh.z > 0.1) * numpy.sqrt(
        (np_mesh.x - cone_origin[0]) ** 2 + (np_mesh.y - cone_origin[1]) ** 2) * numpy.sin(
        numpy.radians(angle_degrees))
    np_mesh.z = np_mesh.z - np_mesh.z.min()

    return np_mesh


def plot(np_mesh: mesh.Mesh, angle_degrees: int | float, cone_origin: numpy.array) -> None:
    """
    Shows a preview of a mesh and an illustration of the cone location and angle.
    :param np_mesh:
    :param angle_degrees:
    :param cone_origin:
    :return:
    """
    logging.info(f"Rendering mesh with {angle_degrees=} and {cone_origin=}")
    mesh_plot = vtkplotlib.mesh_plot(np_mesh)
    vtkplotlib.plot(
        numpy.array([[cone_origin[0], cone_origin[1], -np_mesh.z.max() * 0.1],
                     [cone_origin[0], cone_origin[1], np_mesh.z.max() * 1.1]]),
        color='r',
        fig=mesh_plot.fig)
    for i in range(10):
        vtkplotlib.plot(
            numpy.array([[cone_origin[0], cone_origin[1], i * np_mesh.z.max() / 10],
                         [cone_origin[0] + np_mesh.x.max()* 1.5, cone_origin[1],
                          i * np_mesh.z.max() / 10 + np_mesh.x.max()* 1.5 * numpy.sin(numpy.radians(angle_degrees))]]),
            color='b',
            fig=mesh_plot.fig)
    vtkplotlib.show(block=True, fig=mesh_plot.fig)


def get_distance(point1: tuple, point2: tuple) -> float:
    """
    Calculates the distance between two points in 3D space.
    :param point1:
    :param point2:
    :return:
    """
    distance = numpy.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)
    return distance


def parse_g_gcode(gcode_line: str) -> dict[str:[None | float]]:
    """
    Takes a G command and parses it into a dictionary.

    :param gcode_line:
    :return: an ordered dictionary with the components of the g command. Missing components are set to None.
    """
    if not (gcode_line.strip().startswith('G0') or gcode_line.strip().startswith('G1')):
        msg = f'Invalid command while parsing {gcode_line.strip()}'
        logging.error(msg)
        raise Exception(msg)

    gcode_dict = OrderedDict([('G', None), ('X', None), ('Y', None), ('Z', None), ('E', None), ('F', None)])
    reading = None
    num_str = str()

    for c in gcode_line:
        if c == ';':
            break
        if c in gcode_dict.keys():
            reading = c
            num_str = str()
            continue
        if reading is not None and c != ' ':
            num_str += c
        elif reading is not None and c == ' ':
            gcode_dict[reading] = float(num_str)
    if reading:
        gcode_dict[reading] = float(num_str)

    logging.debug(f"Parsed {gcode_line.strip()} as {gcode_dict}.")
    return gcode_dict


def g_cmd_to_str(gcode_dict: dict[str:[None | float | int]]) -> str:
    """
    Converts a g command dictionary into G code.
    :param gcode_dict:
    :return:
    """
    gcode_str = ''
    for k, v in gcode_dict.items():
        if k == 'G':
            gcode_str += f"{k}{int(v)} "
        elif v is not None:
            gcode_str += f"{k}{v:.7g} "
    logging.debug(f"Converted {gcode_dict=} into {gcode_str=}.")
    return gcode_str


def transform_gcode_line(gcommand: dict,
                         bed_center: tuple,
                         angle_degrees: int | float,
                         previous_location: tuple,
                         first_layer_height: float,
                         cone_origin: numpy.ndarray) -> dict:
    """
    Back-transforms the G command to remove the conic transformation offset that was applied to the original STL file.

    :param gcommand:
    :param bed_center:
    :param angle_degrees:
    :param previous_location:
    :param first_layer_height:
    :param cone_origin:
    :return:
    """
    gcode_dict = gcommand.copy()

    if gcode_dict['X'] is not None or gcode_dict['Y'] is not None or gcode_dict['Z'] is not None:
        tmp_x = float(gcode_dict['X']) if gcode_dict['X'] is not None else previous_location[0]
        tmp_y = float(gcode_dict['Y']) if gcode_dict['Y'] is not None else previous_location[1]
        tmp_z = float(gcode_dict['Z']) if gcode_dict['Z'] is not None else previous_location[2]

        if (tmp_x <= 0) or (tmp_y <= 0):
            return gcode_dict

        travel_distance = get_distance((tmp_x, tmp_y, tmp_z), previous_location)

        tmp_z -= get_distance((tmp_x, tmp_y, 0),
                              (bed_center[0] + cone_origin[0], bed_center[1] + cone_origin[1], 0)) * numpy.tan(
            numpy.radians(angle_degrees))

        gcode_dict['Z'] = max(first_layer_height, tmp_z)

        # adapt extrusion length for new travel distance.
        if (gcode_dict['E'] is not None) and (travel_distance > 0):
            gcode_dict['E'] = gcode_dict['E'] * get_distance((tmp_x, tmp_y, tmp_z), previous_location) / travel_distance
    logging.debug(f"Converted {gcommand=} into {gcode_dict=} when {previous_location=}.")
    return gcode_dict


def cast(type, value):
    try:
        return type(value)
    except ValueError:
        return None


def parse_config(config_file: str) -> dict[str:[str | float | int]]:
    """
    Parses the config file exported by PrusaSlicer and returns it as a dictionary.
    :param config_file:
    :return:
    """
    config = dict()
    with open(config_file) as fh:
        lines = fh.readlines()
    for line in lines:
        l = line.strip()
        # strip comments
        re.sub('#.*', "", l)

        l = l.split('=')
        if len(l) == 2:
            types = [int, float, str]
            for t in types:
                value = cast(t, l[1].strip())
                if value is not None:
                    break
            if l[0].strip() == 'bed_shape':
                bed_shape = numpy.array([list(map(float, x.split('x'))) for x in l[1].strip().split(',')])
                bed_center = (bed_shape[:, 0].max() - bed_shape[:, 0].min()) / 2, (
                        bed_shape[:, 1].max() - bed_shape[:, 1].min()) / 2
                config[l[0].strip()] = bed_shape
                config['bed_center'] = bed_center
            else:
                config[l[0].strip()] = value if value is not None else l[1]
            logger.debug(f'Config property found: {line.strip()} ==> {l[0].strip()}:{config[l[0].strip()]}')
    return config


def get_nozzle_location(move_cmd: dict[str:[int | float]] | str,
                        nozzle_location: tuple[[int | float], [int | float], [int | float]]
                        ) -> tuple[[int | float], [int | float], [int | float]]:
    """
    Utility function to get the location of the nozzle after a G command is executed.

    :param move_cmd: the G command
    :param nozzle_location: the location of the nozzle before the g command is executed
    :return:
    """
    if type(move_cmd) == str:
        move_dict = parse_g_gcode(move_cmd)
    else:
        move_dict: dict = move_cmd
    new_nozzle_location = (move_dict['X'] if move_dict['X'] is not None else nozzle_location[0],
                           move_dict['Y'] if move_dict['Y'] is not None else nozzle_location[1],
                           move_dict['Z'] if move_dict['Z'] is not None else nozzle_location[2])
    return new_nozzle_location


def gcode_transform(gcode_file: str,
                    config_file: str,
                    angle_degrees: int | float,
                    cone_origin: numpy.ndarray) -> list[str]:
    config = parse_config(config_file)
    nozzle_location = (0, 0, 0)
    printing_started = False
    bridging = False
    bridge_segment_count = 0

    with open(gcode_file, "r") as f:
        lines = f.readlines()

    transformed_gcode = []

    for line in tqdm.tqdm(lines, desc="Post-processing gcode"):
        if not (line.startswith('G0') or line.startswith('G1')):
            transformed_gcode.append(line.strip())
            if line.startswith(';TYPE:Overhang'):
                bridging = True
                bridge_segment_count = 0
            elif line.startswith(';TYPE'):
                bridging = False
                bridge_segment_count = 0
            continue
        elif '; go outside print area' in line or '; intro line' in line:
            # Purge line. no transformation needed. from now on, we're printing
            transformed_gcode.append(line.strip())
            printing_started = True
            move_cmd = parse_g_gcode(line)
            nozzle_location = get_nozzle_location(move_cmd=move_cmd, nozzle_location=nozzle_location)
            continue
        elif not printing_started:
            transformed_gcode.append(line.strip())
            move_cmd = parse_g_gcode(line)
            nozzle_location = get_nozzle_location(move_cmd=move_cmd, nozzle_location=nozzle_location)
            continue
        else:
            parsed_g_command = parse_g_gcode(line)
            nozzle_location = get_nozzle_location(move_cmd=parsed_g_command, nozzle_location=nozzle_location)
            move_cmd = transform_gcode_line(parsed_g_command,
                                            bed_center=config['bed_center'],
                                            angle_degrees=angle_degrees,
                                            previous_location=nozzle_location,
                                            first_layer_height=config['first_layer_height'],
                                            cone_origin=cone_origin)

            # If we're bridging add some minor fuzzing to make sure it sticks to the model
            if bridging:
                bridge_segment_count += 1
                if (bridge_segment_count % 2) == 0:
                    if move_cmd['X'] is not None:
                        move_cmd['X'] += numpy.sin(numpy.radians(angle_degrees)) * config['extrusion_width'] * 0.2
                    if move_cmd['Y'] is not None:
                        move_cmd['Y'] += numpy.cos(numpy.radians(angle_degrees)) * config['extrusion_width'] * 0.2

            transformed_gcode.append(g_cmd_to_str(move_cmd))
    return transformed_gcode


def slice_stl(stl_file: str, prusaslicer_path: str, config_file: str, gcode_output: str) -> None:
    """
    Slices the stl file using PrusaSlicer and the config profile provided.

    :param stl_file: the STL file to be sliced
    :param prusaslicer_path: path to `prusa-slicer-console` executable
    :param config_file: the config file used during slicing
    :param gcode_output: the path of the gcode output file
    """

    command = [
        prusaslicer_path,
        '-g', stl_file,
        '--load', config_file,
        '--skirts', '0',  # TODO: make the post-processor clever enough so that we can have a skirt.
        '--output', gcode_output,
        '--infill-overlap', '50%'
    ]
    logger.info(f"Slicing {stl_file}: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        raise Exception("Slicing failed.")


def setup_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input STL file')
    parser.add_argument('-o', '--output', type=str, help='Output gcode file')
    parser.add_argument('--prusa-slicer-path', type=str,
                        default=r'C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe',
                        help='Path to prusa-slicer-console.exe')
    parser.add_argument('--prusa-slicer-config',
                        help="Path to PrusaSlicer config file. Hint: File > Export > Export Config (or CTRL+E).",
                        type=str)
    parser.add_argument('--cone-origin',
                        type=float,
                        nargs=2,
                        default=[0., 0.],
                        help="Origin of the cone, offset from the center of the object.")
    parser.add_argument('--angle',
                        type=float,
                        default=10.0,
                        help='Cone angle in degrees.')
    parser.add_argument('--subdivide-threshold',
                        type=float,
                        default=5.0,
                        help='The minimum acceptable distance between any edges in mm. Used during iterative subdivision')
    parser.add_argument('--max-subdivision-iterations',
                        type=int,
                        default=5,
                        help='The maximum number of iterations to run the subdivision algorithm.')
    parser.add_argument('--preview',
                        action='store_true',
                        default=False,
                        help='Preview the transformed mesh')
    return parser.parse_args()


if __name__ == '__main__':
    args = setup_cli_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError('Input file does not exist.')

    # Load the mesh and make sure it's center is set to (0,0)
    mesh_obj = load_mesh(args.input)
    mesh_obj = reset_origin(mesh_obj)

    # Iteratively subdivide the mesh so that there are no long lines
    i = 0
    while (max_vector := max(
            numpy.sqrt(numpy.sum(numpy.power(mesh_obj.v0 - mesh_obj.v1, 2), axis=1)))) > args.subdivide_threshold:
        i += 1
        logging.info('Subdividing. Max vector length: {:.2f}.'.format(max_vector))
        if i > args.max_subdivision_iterations:
            logging.warning(f"Max iterations for subdivide reached but the maximum vector length is {max_vector}mm.")
            break
        mesh_obj = subdivide(mesh_obj)
    mesh_obj = stl_transform(mesh_obj, angle_degrees=args.angle, cone_origin=numpy.array(args.cone_origin))
    if args.preview:
        plot(np_mesh=mesh_obj, angle_degrees=args.angle, cone_origin=numpy.array(args.cone_origin))

    # TODO: use temporary files or cleanup
    transformed_stl_name = pathlib.Path(args.input).stem + '.transformed.stl'
    intermediate_gcode_name = pathlib.Path(args.input).stem + '.temp.gcode'

    mesh_obj.save(transformed_stl_name)
    slice_stl(transformed_stl_name, args.prusa_slicer_path, args.prusa_slicer_config, intermediate_gcode_name)
    with open(args.output, 'w') as out_f:
        out_f.write('\n'.join(gcode_transform(gcode_file=intermediate_gcode_name,
                                              config_file=args.prusa_slicer_config,
                                              angle_degrees=args.angle,
                                              cone_origin=numpy.array(args.cone_origin))
                              )
                    )
