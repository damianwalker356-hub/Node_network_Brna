import numpy as np
import pytest

from brna.agents import AgentDefinition
from brna.conversation import ConversationMemory
from brna.creatures import (
    ACTIONS,
    CreatureEnvironment,
    run_policy_episode,
    run_q_episode,
)
from brna.evolution import EvolutionTrainer
from brna.reinforcement import QLearner


@pytest.mark.req("R-AGENT-01")
def test_agent_source_round_trips_and_selects_weighted_action():
    source = '''agent "Moss"
rule "Prefer the action with the stronger score."
weight forage: 0.1, 0.9
weight rest: 0.8, 0.1
'''
    agent = AgentDefinition.from_source(source)
    assert AgentDefinition.from_source(agent.to_source()) == agent
    assert agent.choose_action([0.0, 1.0], ["rest", "forage"]) == "forage"


@pytest.mark.req("R-AGENT-02")
def test_agent_source_rejects_executable_python():
    with pytest.raises(ValueError, match="unsupported"):
        AgentDefinition.from_source('''agent "Moss"
import os
''')


@pytest.mark.req("R-CONV-01")
def test_conversation_memory_returns_only_explicitly_taught_answers():
    memory = ConversationMemory()
    memory.teach("What do you eat?", "I eat berries.")
    assert memory.retrieve("What do you eat?") == ("I eat berries.", 1.0)
    assert memory.retrieve("How do I build a spaceship?") is None
    assert ConversationMemory.from_dict(memory.to_dict()).examples == memory.examples


@pytest.mark.req("R-RL-01")
def test_q_learning_applies_discounted_one_step_update():
    learner = QLearner(learning_rate=0.5, discount=0.8, epsilon=0.0)
    learner.q_values["next"] = {"forage": 2.0}
    updated = learner.update("start", "move", 1.0, "next", ["forage"])
    assert updated == pytest.approx(1.3)
    assert learner.choose_action("next", ["rest", "forage"]) == "forage"


@pytest.mark.req("R-EVO-01")
def test_evolution_keeps_elite_and_repeats_with_same_seed():
    first = EvolutionTrainer(population_size=4, elite_fraction=0.25, seed=7)
    second = EvolutionTrainer(population_size=4, elite_fraction=0.25, seed=7)
    population = first.initial_population(3)
    second_population = second.initial_population(3)
    for left, right in zip(population, second_population):
        np.testing.assert_array_equal(left, right)
    fitnesses = [0.0, 1.0, 2.0, 3.0]
    generation_a = first.next_generation(population, fitnesses)
    generation_b = second.next_generation(second_population, fitnesses)
    np.testing.assert_array_equal(generation_a[0], population[3])
    for left, right in zip(generation_a, generation_b):
        np.testing.assert_array_equal(left, right)


@pytest.mark.req("R-CREATURE-01")
def test_policy_and_q_learner_run_in_bounded_creature_environment():
    environment = CreatureEnvironment(
        length=5,
        food_count=1,
        max_steps=20,
        food_positions=[3],
    )
    agent = AgentDefinition(
        "Forager",
        weights={
            "left": (0.0, 0.0, 0.0, 0.0),
            "right": (0.0, 0.0, 0.0, 1.0),
            "eat": (0.0, 1.0, 0.0, 0.0),
            "rest": (-1.0, 0.0, 0.0, 0.0),
        },
    )
    result = run_policy_episode(agent, environment)
    assert result.food_eaten == 1
    assert result.steps <= environment.max_steps

    learner = QLearner(epsilon=1.0, seed=3)
    result = run_q_episode(learner, environment, seed=3)
    assert result.steps <= environment.max_steps
    assert learner.q_values

    evolution = EvolutionTrainer(population_size=4, seed=11)
    population = evolution.initial_population((len(ACTIONS), 4))
    fitnesses = [
        run_policy_episode(
            AgentDefinition(
                f"creature-{index}",
                weights={
                    action: tuple(genome[row])
                    for row, action in enumerate(ACTIONS)
                },
            ),
            environment,
            seed=index,
        ).reward
        for index, genome in enumerate(population)
    ]
    next_generation = evolution.next_generation(population, fitnesses)
    assert len(next_generation) == evolution.population_size
    assert all(genome.shape == (len(ACTIONS), 4) for genome in next_generation)
