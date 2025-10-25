class_name Controller
extends Node

var player: Player


func _enter_tree() -> void:
	player = get_parent() as Player
	assert(player != null, "Controller must be a child of a Player node")

func sample_input() -> void:
	pass
