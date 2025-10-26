class_name Player
extends CharacterBody2D

var gravity: float = 2700.0
var jump_gravity: float = 500.0
var acceleration: float = 10000.0
var friction: float = 2600.0
var air_acceleration: float = 9900.0
var air_friction: float = 24000.0
var h_min_speed: float = 300.0
var h_speed: float = 200.0
var NORMAL_H_SPEED: float = 200.0
var v_speed: float = 3000.0
var jump_impulse: float = 650.0
var jump_buffer_time: float = 0.2
var jump_buffered: bool = false
var flash_impulse: float = 2000.0
var flash_cooldown: float = 3.0
var can_flash: bool = true
var knockback_impulse: float = 300.0

var health: int = 20

var can_punch: bool = true
var can_kick: bool = true

@export var move_state: State
@export var facing_right: bool = false
@export var input: PlayerInput
@export var sprite_sheet: CompressedTexture2D

@onready var controller: Controller = $Controller
@onready var body: CollisionShape2D = $CollisionShape2D
@onready var state_machine: StateMachine = $StateMachine
@onready var sprite: Sprite2D = $Sprite2D
@onready var animation: AnimationPlayer = $AnimationPlayer

@onready var punch_hitbox: Area2D = $PunchHitbox
@onready var kick_hitbox: Area2D = $KickHitbox
@onready var eeg_scanner_hitbox: Area2D = $EEGScan

@onready var hurtbox: Area2D = $Hurtbox

@onready var particles: GPUParticles2D = $GlowParticle
@onready var dirt_particles: GPUParticles2D = $DirtParticle

@onready var dash_sound: AudioStreamPlayer2D = $Dash
@onready var punch_sound: AudioStreamPlayer2D = $Punch
@onready var kick_sound: AudioStreamPlayer2D = $Kick
@onready var damage_sound: AudioStreamPlayer2D = $Damage
@onready var jump_sound: AudioStreamPlayer2D = $Jump


func _ready() -> void:
	sprite.texture = sprite_sheet
	if not facing_right:
		sprite.flip_h = true
		body.scale.x *= -1
		hurtbox.scale.x *= -1
		punch_hitbox.scale.x *= -1
		kick_hitbox.scale.x *= -1
		eeg_scanner_hitbox.scale.x *= -1
	state_machine.initialize()


func _physics_process(delta: float) -> void:
	controller.sample_input()
	state_machine.physics_update(delta)

	if input.move.x < 0 and facing_right:
		facing_right = false
		sprite.flip_h = true
		body.scale.x *= -1
		hurtbox.scale.x *= -1
		punch_hitbox.scale.x *= -1
		kick_hitbox.scale.x *= -1
		eeg_scanner_hitbox.scale.x *= -1
	elif input.move.x > 0 and not facing_right:
		facing_right = true
		sprite.flip_h = false
		body.scale.x *= -1
		hurtbox.scale.x *= -1
		punch_hitbox.scale.x *= -1
		kick_hitbox.scale.x *= -1
		eeg_scanner_hitbox.scale.x *= -1

	if input.punch and can_punch:
		can_punch = false
		punch()

	if input.kick and can_kick:
		can_kick = false
		kick()

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
	var found_EEG = false
	for area in eeg_scanner_hitbox.get_overlapping_areas():
		if area.get_parent().name == "EEGPlayer":
			found_EEG = true
			break
	if found_EEG:
		h_speed = NORMAL_H_SPEED / 1.5
	else:
		h_speed = NORMAL_H_SPEED


#TODO find most efficient way to ground check
func is_grounded() -> bool:
	for cast in $GroundCasts.get_children():
		if cast.is_colliding():
			return true
	return false

func punch() -> void:
	animation.play("punch", -1, 10)
	punch_sound.play()
	var overlapping_areas = punch_hitbox.get_overlapping_areas()
	await get_tree().create_timer(0.25).timeout
	for area in overlapping_areas:
		if area is Hurtbox:
			var hit_player = area.get_parent()
			if hit_player == self:
				continue
			if hit_player.position.x < self.position.x:
				hit_player.take_hit(1, -1)
			else:
				hit_player.take_hit(1, 1)
	await get_tree().create_timer(0.3).timeout
	can_punch = true
	animation.stop()


func kick() -> void:
	animation.play("kick", -1, 15)
	kick_sound.play()
	var overlapping_areas = kick_hitbox.get_overlapping_areas()
	await get_tree().create_timer(0.25).timeout
	for area in overlapping_areas:
		if area is Hurtbox:
			var hit_player = area.get_parent()
			if hit_player == self:
				continue
			if hit_player.position.x < self.position.x:
				hit_player.take_hit(1, -1)
			else:
				hit_player.take_hit(1, 1)
	await get_tree().create_timer(0.3).timeout
	can_kick = true
	animation.stop()


func take_hit(damage: int, direction: int) -> void:
	damage_sound.play()
	self.health -= damage
	h_speed = 13000.0
	self.velocity.x += knockback_impulse * direction
	particles.emitting = true
	state_machine.change_state(move_state)


func handle_flash_step() -> void:
	dash_sound.play()
	h_speed = 3000.0
	can_flash = false
	if input.move.x > 0:
		velocity.x -= flash_impulse
	elif input.move.x < 0:
		velocity.x += flash_impulse
	elif facing_right:
		velocity.x -= flash_impulse
	else:
		velocity.x += flash_impulse
	state_machine.change_state(move_state)
	await get_tree().create_timer(flash_cooldown).timeout
	can_flash = true
