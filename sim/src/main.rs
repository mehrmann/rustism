mod debug;

use bevy::prelude::*;
use bevy::render::camera::ScalingMode;
use bevy::sprite::collide_aabb::collide;
use bevy::window::PresentMode;
use bevy_inspector_egui::{Inspectable};
use lib_neural_network::{LayerTopology, Network};
use rand::prelude::*;
use lib_natural_selection::{Chromosome, GaussianMutation, GeneticAlgorithm, Individual, RouletteWheelSelection, UniformCrossover};
use crate::debug::DebugPlugin;

pub const CLEAR: Color = Color::rgb(0.1, 0.1, 0.1);
pub const ASPECT_RATIO: f32 = 1.0;

#[derive(Resource)]
struct AsciiSheet(Handle<TextureAtlas>);

fn load_ascii(mut commands: Commands,
              assets: Res<AssetServer>,
              mut texture_atlases: ResMut<Assets<TextureAtlas>>) {
    let image = assets.load("Ascii.png");
    let atlas = TextureAtlas::from_grid(
        image,
        Vec2::splat(9.0),
        16, 16,
        Some(Vec2::splat(2.0)), None);

    let atlas_handle = texture_atlases.add(atlas);
    commands.insert_resource(AsciiSheet(atlas_handle));
}

#[derive(Component, Inspectable)]
struct Statistics {
    generation: i32,
    survivors_percentage: f32,
    genetic_variance: f32,
}

impl Statistics {
    fn new() -> Self {
        Self {
            generation: 0,
            survivors_percentage: 0.0,
            genetic_variance: 0.0,
        }
    }
}

#[derive(Component, Inspectable)]
struct KillZone {
    min: f32,
    max: f32,
}

#[derive(Resource)]
struct Config {
    individuals: usize,
    movement_speed: f32,
}

#[derive(Component, Inspectable)]
struct Nizm {
    #[inspectable(ignore)]
    network: Network,
    osc_freq: f32,
    movement: Vec3,
    can_move_left: f32,
    can_move_right: f32,
    can_move_up: f32,
    can_move_down: f32,
    total_movement: f32,
}

#[derive(Component)]
struct Blocking;

impl Nizm {
    fn random(rng: &mut dyn RngCore) -> Self {
        let network = Network::random(
            rng,
            Self::topology(),
        );
        Self {
            network,
            osc_freq: 1.0,
            movement: Vec3::ZERO,
            can_move_left: 1.0,
            can_move_right: 1.0,
            can_move_up: 1.0,
            can_move_down: 1.0,
            total_movement: 0.0,
        }
    }

    fn reset(&mut self) {
        self.osc_freq = 1.0;
        self.movement = Vec3::ZERO;
        self.total_movement = 0.0;
    }

    fn topology() -> &'static [LayerTopology] {
        &[
            LayerTopology { neurons: 11 },
            LayerTopology { neurons: 24 },
            LayerTopology { neurons: 5 },
        ]
    }
}

#[derive(Component)]
struct Position(Transform);

#[derive(Resource)]
struct EvolutionTimer(Timer);

struct NizmIndividual {
    chromosome: Chromosome,
    fitness: f32,
}

impl Individual for NizmIndividual {
    fn create(chromosome: Chromosome) -> Self {
        Self {
            chromosome,
            fitness: 0.0,
        }
    }

    fn fitness(&self) -> f32 {
        self.fitness
    }

    fn chromosome(&self) -> &Chromosome {
        &self.chromosome
    }
}

fn chromosome_to_color(chromosome: &Chromosome) -> Color {
    let hash: i32 = chromosome.iter().fold(0, |acc, v| (acc.wrapping_mul(23).wrapping_add((v * 100.0) as i32)));

    let float1 = (hash as f64 * 23.32).sin() as f32;
    let float2 = (hash as f64 * 73261.2).sin() as f32;
    let float3 = (hash as f64 * 32132.21).sin() as f32;
    return Color::rgb(float1, float2, float3);
}

fn update_statistics(timer: Res<EvolutionTimer>,
                     mut query: Query<(&mut Text, &Statistics)>) {
    for (mut text, statistics) in query.iter_mut() {
        let generation = statistics.generation;
        let survivor_percentage = statistics.survivors_percentage;
        let time_left_in_generation = timer.0.remaining().as_secs_f32();
        text.sections[0].value = format!("Time: {time_left_in_generation:.1}s\nGeneration: {generation}\nPercentage: {survivor_percentage:.2}");
    }
}


fn evolution(time: Res<Time>,
             config: Res<Config>,
             mut timer: ResMut<EvolutionTimer>,
             mut query: Query<(Entity, &mut Nizm, &mut Transform, &mut TextureAtlasSprite), Without<KillZone>>,
             mut statistics: Query<&mut Statistics>,
             mut killzone: Query<(&mut KillZone, &mut Transform)>) {
    if (timer.0.tick(time.delta())).just_finished() {
        let (mut killzone, mut killzonetransform) = killzone.get_single_mut().expect("need killzone");

        let mut survivors = Vec::new();

        for (_entity, brain, transform, _sprite) in query.iter_mut() {
            survivors.push(NizmIndividual {
                chromosome: brain.network.data().collect(),
                fitness: if transform.translation.x < killzone.max && transform.translation.x > killzone.min { 0.0 } else { transform.translation.x.abs() + 1.0 + brain.total_movement },
            });
        }

        let ga = GeneticAlgorithm::new(
            RouletteWheelSelection::new(),
            UniformCrossover::default(),
            GaussianMutation::new(0.3, 0.5));

        let mut rng = thread_rng();
        let offspring = ga.evolve(&mut rng, &survivors);

        for ((_entity, mut brain, mut transform, mut sprite), child) in query.iter_mut().zip(offspring) {
            brain.network = Network::from_data(Nizm::topology(), child.chromosome.clone());
            brain.reset();
            transform.translation = Vec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 900.0);
            sprite.color = chromosome_to_color(child.chromosome());
        }

        let mut stats = statistics.get_single_mut().expect("Stats");
        stats.generation = stats.generation + 1;
        stats.survivors_percentage = survivors.iter().filter(|s| s.fitness > 0.0).count() as f32 / config.individuals as f32;

        let x = killzone_pos();
        killzone.min = x;
        killzone.max = x + 1.0;
        killzonetransform.translation.x = x + 0.5;
    }
}

fn check_collision(target: Vec3, other: Vec3) -> bool {
    let collision = collide(
        target,
        Vec2::splat(0.03) * 0.8,
        other,
        Vec2::splat(0.03),
    );
    collision.is_some()
}

fn check_if_can_move(time: Res<Time>,
                     config: Res<Config>,
                     mut query: Query<(&mut Nizm, &Transform)>) {
    let left = Vec3::new(-1.0, 0.0, 0.0) * 0.8 * time.delta().as_secs_f32() * config.movement_speed;
    let right = Vec3::new(1.0, 0.0, 0.0) * 0.8 * time.delta().as_secs_f32() * config.movement_speed;
    let up = Vec3::new(0.0, -1.0, 0.0) * 0.8 * time.delta().as_secs_f32() * config.movement_speed;
    let down = Vec3::new(0.0, 1.0, 0.0) * 0.8 * time.delta().as_secs_f32() * config.movement_speed;

    let mut combinations = query.iter_combinations_mut();
    while let Some([(mut nizm, transform), (_, other)]) = combinations.fetch_next() {
        nizm.can_move_left = if check_collision(transform.translation + left, other.translation.clone()) { 1.0 } else { 0.0 };
        nizm.can_move_right = if check_collision(transform.translation + right, other.translation.clone()) { 1.0 } else { 0.0 };
        nizm.can_move_up = if check_collision(transform.translation + up, other.translation.clone()) { 1.0 } else { 0.0 };
        nizm.can_move_down = if check_collision(transform.translation + down, other.translation.clone()) { 1.0 } else { 0.0 };
    }
}

fn move_individuals(time: Res<Time>,
                    config: Res<Config>,
                    mut query: Query<(Entity, &mut Nizm)>,
                    mut transforms: Query<&mut Transform, With<Blocking>>) {
    for (entity, mut nizm) in query.iter_mut() {
        let translation = transforms.get_mut(entity).expect("WTF").translation;

        let mut movement = nizm.movement * time.delta().as_secs_f32() * config.movement_speed;
        let target = translation + movement;

        // if (target.x < -1.0) { translation.x += 2.0; }
        // if (target.x > 1.0) { translation.x -= 2.0; }
        // if (target.y < -1.0) { translation.y += 2.0; }
        // if (target.y > 1.0) { translation.y -= 2.0; }

        if target.x.abs() > 1.0 { movement.x = 0.0 }
        if target.y.abs() > 1.0 { movement.y = 0.0 }

        if !transforms.iter().any(|t| {
            if translation != t.translation {
                check_collision(translation + movement, t.translation.clone())
            } else {
                false
            }
        }) {
            transforms.get_mut(entity).expect("WTF").translation = translation + movement;
            nizm.total_movement += movement.length();
            nizm.movement = movement;
        } else {
            nizm.movement = Vec3::ZERO;
        }
    }
}

fn make_individuals_think(timer: Res<EvolutionTimer>,
                          mut transforms: Query<&mut Transform>,
                          mut nizms: Query<(Entity, &mut Nizm)>,
                          killzone: Query<&KillZone>, ) {
    let killzone = killzone.get_single().expect("need killzone");

    for (entity, mut nizm) in nizms.iter_mut() {
        let translation = transforms.get_mut(entity).expect("WTF").translation;
        let remaining = timer.0.elapsed_secs() / timer.0.duration().as_secs_f32();
        let osc = (nizm.osc_freq * remaining * 3.14 * 2.0).sin();
        let result = nizm.network.propagate(vec![
            translation.x,
            translation.y,
            remaining,
            osc,
            nizm.movement.x,
            nizm.movement.y,
            nizm.can_move_left,
            nizm.can_move_right,
            nizm.can_move_up,
            nizm.can_move_down,
            if killzone.min < 0.0 { -1.0 } else { 1.0 },
        ]);

        let movement = Vec3::new(
            result[0].clamp(0.0, 1.0) - result[1].clamp(0.0, 1.0),
            result[2].clamp(0.0, 1.0) - result[3].clamp(0.0, 1.0),
            0.0).normalize_or_zero();
        nizm.movement = movement;
        nizm.osc_freq = result[4];


        //transforms.get_mut(entity).expect("WTF").translation = target;
    }
}

fn add_statistics_text(mut commands: Commands, assets: Res<AssetServer>) {
    commands.spawn((
        // Create a TextBundle that has a Text with a list of sections.
        TextBundle::from_sections([
            TextSection::new(
                "XXX",
                TextStyle {
                    font: assets.load("fonts/FiraMono-Medium.ttf"),
                    font_size: 20.0,
                    color: Color::WHITE,
                },
            )
        ])
            .with_text_alignment(TextAlignment::TOP_LEFT)
            // Set the style of the TextBundle itself.
            .with_style(Style {
                position_type: PositionType::Absolute,
                position: UiRect {
                    bottom: Val::Px(5.0),
                    left: Val::Px(15.0),
                    ..default()
                },
                ..default()
            }),
        Statistics::new(),
    ));
}

fn init_killzone(mut commands: Commands) {
    let x = killzone_pos();

    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::rgb(0.2, 0.0, 0.0),
                custom_size: Some(Vec2::new(1.0, 2.0)),
                ..default()
            },
            transform: Transform::from_translation(Vec3::new(x + 0.5, 0.0, 10.0)),
            ..default()
        },
        KillZone { min: x, max: x + 1.0 }
    ));
}

fn killzone_pos() -> f32 {
    let mut rng = thread_rng();

    let x = if rng.gen_bool(0.5) {
        -1.0
    } else {
        0.0
    };
    x
}

fn add_individuals(config: Res<Config>, ascii: Res<AsciiSheet>, mut commands: Commands) {
    let mut rng = thread_rng();

    for i in 0..config.individuals {
        let mut sprite = TextureAtlasSprite::new(1);
        sprite.custom_size = Some(Vec2::splat(0.03));

        let network = Network::random(
            &mut rng,
            Nizm::topology(),
        );

        sprite.color = chromosome_to_color(&network.data().collect());

        commands.spawn((
            SpriteSheetBundle {
                sprite: sprite,
                texture_atlas: ascii.0.clone(),
                transform: Transform {
                    translation: Vec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 900.0),
                    ..default()
                },
                ..default()
            },
            Name::new(format!("nizm_{i}")),
            Nizm::random(&mut rng),
            Blocking
        ));
    }
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn(Camera2dBundle {
        projection: OrthographicProjection {
            top: 1.0,
            bottom: -1.0,
            right: 1.0 * ASPECT_RATIO,
            left: -1.0 * ASPECT_RATIO,
            scaling_mode: ScalingMode::None,
            scale: 1.,
            ..default()
        },
        ..default()
    });
}

fn main() {
    let height: f32 = 800.0;

    App::new()
        .insert_resource(ClearColor(CLEAR))
        .insert_resource(Config { individuals: 128, movement_speed: 0.5 })
        .insert_resource(EvolutionTimer(Timer::from_seconds(8.0, TimerMode::Repeating)))
        .add_startup_system(spawn_camera)
        .add_startup_system(add_individuals)
        .add_startup_system(add_statistics_text)
        .add_startup_system(init_killzone)
        .add_startup_system_to_stage(StartupStage::PreStartup, load_ascii)
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            window: WindowDescriptor {
                title: "Rustism".to_string(),
                width: height * ASPECT_RATIO,
                height: height,
                present_mode: PresentMode::AutoNoVsync,
                ..default()
            },
            ..default()
        }))
        .add_system(evolution)
        .add_system(check_if_can_move.before(make_individuals_think))
        .add_system(make_individuals_think.before(move_individuals))
        .add_system(move_individuals)
        .add_system(update_statistics)
        .add_plugin(DebugPlugin)
        .run();

}
