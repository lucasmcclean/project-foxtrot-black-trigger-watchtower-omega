class_name LocalController
extends Controller

func sample_input() -> void:
	if player.input == null:
		return

	player.input.move.x = (
		Input.get_action_strength("move_right") - Input.get_action_strength("move_left")
	)
	player.input.move.y = (
		Input.get_action_strength("move_down") - Input.get_action_strength("move_up")
	)

	player.input.jump = Input.is_action_just_pressed("jump")
	player.input.punch = Input.is_action_just_pressed("punch")
	player.input.kick = Input.is_action_just_pressed("kick")
	player.input.flash_step = Input.is_action_just_pressed("flash_step")
	player.input.crouch = Input.is_action_just_pressed("crouch")
	
