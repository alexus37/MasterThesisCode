#version 330 core

in vec2 final_text_coord;

out vec4 frag_color;

uniform sampler2D the_texture;

void main() {
    frag_color = vec4(1, 0, 0, 1);
}