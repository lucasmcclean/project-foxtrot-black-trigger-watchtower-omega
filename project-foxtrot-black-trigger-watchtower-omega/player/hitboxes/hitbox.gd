class_name Hitbox
extends Area2D

enum HitboxType {
	PUNCH,
	KICK
}

@export var type: HitboxType
