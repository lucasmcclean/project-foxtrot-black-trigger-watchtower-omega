extends PlayerState

@export var idle_state: State
@export var jump_state: State
@export var move_state: State


func enter() -> void:
	super()
	if not player.animation.is_playing():
		player.animation.play("fall")


func update(_delta: float):
	if not player.animation.is_playing():
		player.animation.play("fall")


func physics_update(delta: float) -> void:
	player.velocity.y += player.gravity*delta
	if (player.input.move.x != 0) and (player.velocity.x*player.input.move.x > 0):
		player.velocity.x += player.input.move.x*player.air_acceleration*delta
	elif player.input.move.x != 0:
		player.velocity.x += player.input.move.x*player.air_friction*delta
	else:
		player.velocity.x = move_toward(player.velocity.x, 0, player.air_friction*delta)
	player.velocity.x = clamp(player.velocity.x, -player.h_speed, player.h_speed)
	player.velocity.y = clamp(player.velocity.y, -player.v_speed, player.v_speed)
	player.move_and_slide()
	
	if player.input.jump:
		player.jump_buffered = true
		get_tree().create_timer(player.jump_buffer_time).timeout.connect(func(): player.jump_buffered = false)
	
	if player.is_grounded():
		player.dirt_particles.emitting = true
		if player.jump_buffered:
			state_machine.change_state(jump_state)
		elif !is_zero_approx(player.velocity.x):
			state_machine.change_state(move_state)
		else:
			state_machine.change_state(idle_state)
