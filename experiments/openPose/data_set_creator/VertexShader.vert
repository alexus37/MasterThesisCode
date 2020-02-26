#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 text_coord;

out vec2 final_text_coord;

uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    final_text_coord = text_coord;
}