from pathlib import Path
import math

import wgpu
from rendercanvas.glfw import RenderCanvas, loop
import graphics_math as gm

from camera3d import Camera3D
from pointlight import PointLight
from shader import Shader
from pipeline import Pipeline
from phongmaterial import PhongMaterial
from texture import Texture
from sampler import Sampler
from textureset import TextureSet
from transform import Transform
from node import Node
from scene import Scene
from renderer import Renderer
from cube import Cube
from sphere import Sphere

BASE = Path(__file__).resolve().parent.parent
FOG_COLOR = (1.0, 1.0, 1.0, 1.0)
fog_enabled = False
normal_enabled = True


def initialize(device, target_format):
    global camera, scene, shader, fog_enabled, normal_enabled
    fog_enabled = False
    normal_enabled = True

    camera = Camera3D(5.0, 4.8, 7.5)
    camera.set_center(0.0, 1.55, 0.0)
    camera.set_angle(42.0)
    camera.set_z_planes(0.1, 100.0)
    camera.create_arcball().attach(canvas)


    position = gm.vec3(-2.5, 6.0, 3.0)
    target = gm.vec3(0.0, 1.3, 0.0)
    light = PointLight(position[0], position[1], position[2], space="world",
                       spot_direction=gm.normalize(target - position),
                       spot_cutoff=math.cos(math.radians(22.0)),
                       spot_exponent=4.0)
    light.set_ambient(0.18, 0.18, 0.18)
    light.set_diffuse(0.82, 0.82, 0.82)

    shader = Shader(device, str(BASE / "shaders/cena.wgsl"),
                    light=light, space="world")

    shader.set_vertex_buffers([
        {"array_stride": 12, "step_mode": "vertex", "attributes": [
            {"format": "float32x3", "offset": 0, "var_name": "coord"}]},
        {"array_stride": 12, "step_mode": "vertex", "attributes": [
            {"format": "float32x3", "offset": 0, "var_name": "normal"}]},
        {"array_stride": 12, "step_mode": "vertex", "attributes": [
            {"format": "float32x3", "offset": 0, "var_name": "tangent"}]},
        {"array_stride": 8, "step_mode": "vertex", "attributes": [
            {"format": "float32x2", "offset": 0, "var_name": "texcoord"}]},
    ])
    pipeline = Pipeline(shader, target_format, primitive={"cull_mode": "back"})
    shader.set_value("fog_density", 0.075)
    shader.set_value("fog_color", gm.vec4(*FOG_COLOR))
    update_effects()

    branco = PhongMaterial(0.82, 0.82, 0.82)
    amarelo = PhongMaterial(0.95, 0.75, 0.20)
    terra = PhongMaterial(1.0, 1.0, 1.0)
    vermelho = PhongMaterial(0.95, 0.16, 0.18)
    branco.set_specular(0.15, 0.15, 0.15)
    amarelo.set_specular(0.20, 0.20, 0.20)
    for material in [branco, amarelo, terra, vermelho]:
        shader.add_material(material)


    white = Texture(device, "decal_texture", None, texel=gm.vec3(1, 1, 1))
    paper = Texture(device, "decal_texture", str(BASE / "images/paper.jpg"))
    earth = Texture(device, "decal_texture", str(BASE / "images/earth.jpg"))

    neutral = Texture(device, "normal_texture", None,
                      texel=gm.vec3(0.5, 0.5, 1.0), format="rgba8unorm")
    golf = Texture(device, "normal_texture", str(BASE / "images/golfball.png"),
                   format="rgba8unorm")
    decal_sampler = Sampler(device, "decal_sampler")
    normal_sampler = Sampler(device, "normal_sampler")
    plain_set = TextureSet([white, decal_sampler, neutral, normal_sampler])
    paper_set = TextureSet([paper, decal_sampler, neutral, normal_sampler])
    earth_set = TextureSet([earth, decal_sampler, neutral, normal_sampler])
    golf_set = TextureSet([white, decal_sampler, golf, normal_sampler])
    for textures in [plain_set, paper_set, earth_set, golf_set]:
        shader.add_texture_set(textures)

    cube = Cube(device)
    sphere = Sphere(device)


    trf = Transform()
    trf.translate(0.0, 1.8, 0.0)
    trf.scale(4.0, 0.2, 3.0)
    mesa = Node(apps=[branco], nodes=[Node(trf=trf, shps=[cube])])
    for x, z in [(-1.72, -1.22), (1.72, -1.22),
                 (-1.72, 1.22), (1.72, 1.22)]:
        trf = Transform()
        trf.translate(x, 0.0, z)
        trf.scale(0.22, 1.8, 0.22)
        mesa.add_node(Node(trf=trf, shps=[cube]))

    trf_grupo = Transform()
    trf_grupo.translate(-0.70, 2.0, 0.25)
    trf_bloco = Transform()
    trf_bloco.scale(1.15, 0.35, 0.90)
    bloco = Node(trf=trf_bloco, apps=[amarelo, paper_set], shps=[cube])
    trf_terra = Transform()
    trf_terra.translate(0.0, 0.35 + 0.32, 0.0)
    trf_terra.rotate(-140.0, 0.0, 1.0, 0.0)
    trf_terra.scale(0.32, 0.32, 0.32)
    esfera_terra = Node(trf=trf_terra, apps=[terra, earth_set], shps=[sphere])
    grupo = Node(trf=trf_grupo, nodes=[bloco, esfera_terra])

    trf_golfe = Transform()
    trf_golfe.translate(0.90, 2.0 + 0.43, -0.45)
    trf_golfe.scale(0.43, 0.43, 0.43)
    esfera_golfe = Node(trf=trf_golfe, apps=[vermelho, golf_set], shps=[sphere])


    root = Node(pipeline=pipeline, apps=[plain_set],
                nodes=[mesa, grupo, esfera_golfe])
    scene = Scene(root)


def update_effects():
    shader.set_value("fog_enabled", float(fog_enabled))
    shader.set_value("normal_enabled", float(normal_enabled))


def draw():
    renderer.render(context.get_current_texture(), scene, camera)


def on_key(event):
    global fog_enabled, normal_enabled
    key = event["key"].lower()
    if key == "q":
        canvas.close()
    elif key == "f":
        fog_enabled = not fog_enabled
        update_effects()
        print("Fog:", "ligado" if fog_enabled else "desligado")
    elif key == "r":
        normal_enabled = not normal_enabled
        update_effects()
        print("Rugosidade:", "ligada" if normal_enabled else "desligada")
    elif key == "home":

        camera.get_arcball().mat = gm.mat4(1.0)
        print("Camera na vista inicial")


def main():
    global canvas, context, renderer
    canvas = RenderCanvas(size=(960, 720), title="Projeto 2 - Mesa e objetos",
                          update_mode="continuous", max_fps=60)
    adapter = wgpu.gpu.request_adapter_sync()
    device = adapter.request_device_sync()
    context = canvas.get_context("wgpu")
    target_format = context.get_preferred_format(device.adapter)
    context.configure(device=device, format=target_format)
    renderer = Renderer(device, depth_test=True, clear_value=FOG_COLOR)
    initialize(device, target_format)
    canvas.add_event_handler(on_key, "key_down")
    print("Arraste o mouse: camera | F: fog | R: rugosidade | Home: vista inicial | Q: sair")
    canvas.request_draw(draw)
    loop.run()


if __name__ == "__main__":
    main()
