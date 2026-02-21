"""Semantic Kernel orchestrator for multi-step task handling."""

from typing import Optional
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings

from src.sk_integration.plugins import ChatPlugins, load_chat_plugins
from src.sk_integration.memory_manager import ConversationMemory


class SKOrchestrator:
    """Orchestrates multi-step tasks using Semantic Kernel."""

    def __init__(self, plugins: Optional[ChatPlugins] = None, 
                 enable_memory: bool = True,
                 model_type: str = "auto"):
        """
        Initialize SK orchestrator.
        
        Args:
            plugins: ChatPlugins instance (loaded automatically if None)
            enable_memory: Whether to enable conversation memory
            model_type: Type of model to load
        """
        self.kernel = Kernel()
        
        # Load plugins if not provided
        if plugins is None:
            plugins = load_chat_plugins(model_type=model_type)
        
        self.plugins = plugins
        self.memory = ConversationMemory() if enable_memory else None
        
        # Register plugins with kernel
        self._register_plugins()

    def _register_plugins(self):
        """Register chat plugins with the kernel."""
        # Add the whole plugin class to kernel
        self.kernel.add_plugin(self.plugins, plugin_name="chat")

    def execute_task(self, user_input: str, context: Optional[str] = None) -> tuple[str, dict]:
        """
        Execute a task with optional context enrichment.
        
        Args:
            user_input: User's input/question
            context: Optional additional context
            
        Returns:
            Tuple of (response, metadata)
        """
        # Enrich prompt with conversation history if available
        prompt = user_input
        if self.memory:
            history_context = self.memory.get_context()
            if history_context:
                prompt = f"{history_context}\n\nNew question: {user_input}"

        if context:
            prompt = f"{context}\n\n{prompt}"

        # Execute generation
        response = self.plugins.generate_response(prompt, use_instruction_format=True)
        
        # Store in memory if enabled
        metadata = {}
        if self.memory:
            self.memory.add_exchange(user_input, response, metadata)
            metadata["memory_summary"] = self.memory.get_summary()

        metadata["model_info"] = self.plugins.get_model_info()
        
        return response, metadata

    def execute_with_refinement(self, user_input: str, max_iterations: int = 2) -> str:
        """
        Execute a task with iterative refinement.
        
        Args:
            user_input: Initial user input
            max_iterations: Maximum refinement iterations
            
        Returns:
            Refined response
        """
        response = user_input
        
        for i in range(max_iterations):
            # Generate response for current iteration
            refined_response, _ = self.execute_task(response)
            
            if i < max_iterations - 1:
                # Ask model to refine the response
                refinement_prompt = (
                    f"The following is a response that could be improved. "
                    f"Please provide a more detailed and clear version:\n\n{refined_response}"
                )
                response = refinement_prompt
            else:
                response = refined_response

        return response

    def execute_multi_step(self, steps: list[str]) -> list[tuple[str, dict]]:
        """
        Execute multiple steps in sequence.
        
        Args:
            steps: List of prompts/steps to execute
            
        Returns:
            List of (response, metadata) tuples
        """
        results = []
        for step in steps:
            response, metadata = self.execute_task(step)
            results.append((response, metadata))
        
        return results

    def get_conversation_summary(self) -> dict:
        """Get summary of conversation if memory is enabled."""
        if not self.memory:
            return {"error": "Memory not enabled"}
        
        return self.memory.get_summary()

    def clear_memory(self):
        """Clear conversation history."""
        if self.memory:
            self.memory.clear()

    def get_full_history(self) -> list:
        """Get full conversation history."""
        if not self.memory:
            return []
        
        return self.memory.get_full_history()

    def process_command(self, command: str) -> str:
        """
        Process system commands.
        
        Args:
            command: Command string (e.g., "settings", "info", "clear")
            
        Returns:
            Command result
        """
        if command == "info":
            return self.plugins.get_model_info()
        elif command == "settings":
            return self.plugins.get_settings()
        elif command == "memory":
            return str(self.get_conversation_summary()) if self.memory else "Memory not enabled"
        elif command == "clear":
            self.clear_memory()
            return "Conversation history cleared"
        else:
            return f"Unknown command: {command}"
