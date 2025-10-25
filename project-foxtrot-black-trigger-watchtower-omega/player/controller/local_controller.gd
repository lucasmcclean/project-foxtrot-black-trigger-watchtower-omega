class_name LocalController
extends Node

@export var player_input: PlayerInput

func sample_input() -> void:
	if player_input == null:
		return

	player_input.move.x = (
		Input.get_action_strength("move_right") - Input.get_action_strength("move_left")
	)
	player_input.move.y = (
		Input.get_action_strength("move_down") - Input.get_action_strength("move_up")
	)

	player_input.jump = Input.is_action_just_pressed("jump")
	player_input.punch = Input.is_action_just_pressed("punch")
	player_input.kick = Input.is_action_just_pressed("kick")
	player_input.flash_step = Input.is_action_just_pressed("flash_step")
