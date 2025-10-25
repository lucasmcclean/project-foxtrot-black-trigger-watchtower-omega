class_name Player
extends CharacterBody2D

var gravity: float = 8000.0
var jump_gravity: float = 800.0
var acceleration: float = 10000.0
var friction: float = 25000.0
var air_acceleration: float = 9900.0
var air_friction: float = 24000.0
var h_min_speed: float = 300.0
var h_speed: float = 300.0
var v_speed: float = 3000.0
var jump_impulse: float = 1500.0
var flash_impulse: float = 4000.0
var flash_cooldown: float = 3.0
var can_flash: bool = true

var health: int = 20

var can_punch: bool = true
var can_kick: bool = true

@export var  move_state: State
@export var facing_right: bool = false
@export var input: PlayerInput

@onready var controller: Controller = $Controller
@onready var state_machine: StateMachine = $StateMachine
@onready var sprite: Sprite2D = $Sprite2D

@onready var punch_hitbox: Area2D = $PunchHitbox
@onready var kick_hitbox: Area2D = $KickHitbox

@onready var hurtbox: Area2D = $Hurtbox


func _ready() -> void:
	if not facing_right:
		sprite.flip_h = true
		punch_hitbox.scale.x *= -1
		kick_hitbox.scale.x *= -1
	state_machine.initialize()


func _physics_process(delta: float) -> void:
	controller.sample_input()
	state_machine.physics_update(delta)

	if input.move.x < 0 and facing_right:
		facing_right = false
		sprite.flip_h = true
		punch_hitbox.scale.x *= -1
		kick_hitbox.scale.x *= -1
	elif input.move.x > 0 and not facing_right:
		facing_right = true
		sprite.flip_h = false
		punch_hitbox.scale.x *= -1
		kick_hitbox.scale.x *= -1
		
	if input.punch and can_punch :
		can_punch = false
		perform_attack(punch_hitbox)
		
	if input.kick:
		can_kick = false
		perform_attack(kick_hitbox)
	
	if input.flash_step and can_flash and is_grounded():
		# Courtesy of Ishfaq
		hurtbox.monitorable = false
		hurtbox.monitoring = false
		handle_flash_step()
	else:
		hurtbox.monitorable = true
		hurtbox.monitoring = true


func _process(delta: float) -> void:
	state_machine.update(delta)


#TODO find most efficient way to ground check
func is_grounded() -> bool:
	for cast in $GroundCasts.get_children():
		if cast.is_colliding():
			return true
	return false


func perform_attack(hitbox: Area2D) -> void:
	var overlapping_areas = hitbox.get_overlapping_areas()
	for area in overlapping_areas:
		if area is Hurtbox:
			area.get_parent().health -= 1
			print("attacking")
	await get_tree().create_timer(0.5).timeout
	can_punch = true
	can_kick = true



func handle_flash_step() -> void:
	if(input.move.x == 0):
		return
	h_speed = 30000.0
	can_flash = false
	if(input.move.x > 0):
		velocity.x -= flash_impulse
	else:
		velocity.x +=  flash_impulse
		velocity.x += flash_impulse
	state_machine.change_state(move_state)
	await get_tree().create_timer(flash_cooldown).timeout
	can_flash = true
