extends PlayerState

@export var fall_state: State
@export var jump_state: State
@export var idle_state: State


func enter() -> void:
	super()
	if not player.animation.is_playing():
		player.animation.play("walk")


func update(_delta: float):
	if not player.animation.is_playing():
		player.animation.play("fall")


func physics_update(delta: float) -> void:
	if player.input.jump and player.is_grounded():
		state_machine.change_state(jump_state)
		return

	player.velocity.y += player.gravity*delta

	if player.input.move.x != 0:
		player.velocity.x = move_toward(player.velocity.x, player.input.move.x * player.h_speed, player.acceleration * delta)
	else:
		player.velocity.x = move_toward(player.velocity.x, 0, player.friction * delta)

	if(player.h_speed > player.h_min_speed):
		player.h_speed -= (player.h_speed - player.h_min_speed)/2
		if(player.h_speed <= 100):
			player.h_speed = player.h_min_speed
	player.velocity.x = clamp(player.velocity.x, -player.h_speed, player.h_speed)
	player.move_and_slide()

	if player.velocity.y > 0:
		state_machine.change_state(fall_state)
	elif is_zero_approx(player.velocity.x):
		state_machine.change_state(idle_state)
