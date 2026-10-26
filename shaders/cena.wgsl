struct Matrix {
  vertex: mat4x4<f32>,
  normal: mat4x4<f32>,
}
@group(1) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,
  camera_position: vec4<f32>,
  light_position: vec4<f32>,
  light_ambient: vec3<f32>,
  light_diffuse: vec3<f32>,
  light_specular: vec3<f32>,
  spot_direction: vec3<f32>,
  spot_cutoff: f32,
  spot_exponent: f32,
  fog_enabled: f32,
  fog_density: f32,
  normal_enabled: f32,
  fog_color: vec4<f32>,
}
@group(0) @binding(0) var<uniform> global: Global;

struct Material {
  base_color: vec3<f32>,
  opacity: f32,
  specular_color: vec3<f32>,
  shininess: f32,
}
@group(2) @binding(0) var<uniform> material: Material;
@group(3) @binding(0) var decal_texture: texture_2d<f32>;
@group(3) @binding(1) var decal_sampler: sampler;
@group(3) @binding(2) var normal_texture: texture_2d<f32>;
@group(3) @binding(3) var normal_sampler: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) light: vec3<f32>,
  @location(1) view: vec3<f32>,
  @location(2) texcoord: vec2<f32>,
  @location(3) world_position: vec3<f32>,
}

@vertex
fn vs_main(@builtin(instance_index) i: u32,
           @location(0) coord: vec3<f32>,
           @location(1) normal: vec3<f32>,
           @location(2) tangent: vec3<f32>,
           @location(3) texcoord: vec2<f32>) -> VertexOutput {
  let m = matrix[i];
  let p = (m.vertex * vec4<f32>(coord, 1.0)).xyz;


  let n = normalize((m.normal * vec4<f32>(normal, 0.0)).xyz);
  let t = normalize((m.normal * vec4<f32>(tangent, 0.0)).xyz);
  let b = cross(n, t);
  let to_tangent = mat3x3<f32>(
    vec3<f32>(t.x, b.x, n.x),
    vec3<f32>(t.y, b.y, n.y),
    vec3<f32>(t.z, b.z, n.z)
  );

  var out: VertexOutput;
  out.clip_position = global.projection * vec4<f32>(p, 1.0);
  out.world_position = p;
  out.texcoord = texcoord;
  out.light = to_tangent * (global.light_position.xyz - p);
  out.view = to_tangent * (global.camera_position.xyz - p);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

  let decal = textureSample(decal_texture, decal_sampler, in.texcoord).rgb;
  let mapped = textureSample(normal_texture, normal_sampler, in.texcoord).rgb;
  var normal = vec3<f32>(0.0, 0.0, 1.0);
  if (global.normal_enabled > 0.5) {
    normal = normalize(2.0 * mapped - vec3<f32>(1.0));
  }
  let light = normalize(in.light);
  let view = normalize(in.view);


  let light_world = normalize(global.light_position.xyz - in.world_position);
  let cosine = dot(-light_world, normalize(global.spot_direction));
  var spot = 0.0;
  if (cosine >= global.spot_cutoff) {
    spot = pow(max(cosine, 0.0), global.spot_exponent);
  }

  var color = material.base_color * decal * global.light_ambient;
  let ndotl = dot(normal, light);

  if (ndotl > 0.0 && light.z > 0.0) {
    color += spot * material.base_color * decal * global.light_diffuse * ndotl;
    let refl = normalize(reflect(-light, normal));
    let specular = pow(max(dot(refl, view), 0.0), material.shininess);
    color += spot * material.specular_color * global.light_specular * specular;
  }


  if (global.fog_enabled > 0.5) {
    let distance = length(global.camera_position.xyz - in.world_position);
    let factor = exp(-pow(global.fog_density * distance, 2.0));
    color = factor * color + (1.0 - factor) * global.fog_color.rgb;
  }
  return vec4<f32>(color, material.opacity);
}
