from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PayloadSchemaType,
    CreateCollection,
    FieldCondition,
    Filter,
    MatchValue
)
from rich.console import Console
from typing import Optional

console = Console()
load_dotenv()
# Collection names
RESOURCES_COLLECTION = "mental_health_resources"
MEMORY_COLLECTION = "user_memory"
USER_PROFILE_COLLECTION = "user_profiles"
MULTIMODAL_COLLECTION = "multimodal_data"

class QdrantInitializer:
    """Initialize Qdrant Cloud collections with proper configuration"""
    
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize Qdrant client for cloud instance
        
        Args:
            url: Qdrant cloud URL (e.g., 'https://xxx.aws.cloud.qdrant.io:6333')
            api_key: Qdrant cloud API key
        """
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        if not self.url or not self.api_key:
            raise ValueError("Qdrant URL and API key must be provided or set as environment variables")
        
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=60
        )
        
        console.print(f"[green]✓[/green] Connected to Qdrant Cloud at {self.url}")
    
    def create_collection_if_not_exists(
        self, 
        collection_name: str, 
        vector_size: int = 384,
        distance: Distance = Distance.COSINE,
        description: str = ""
    ) -> bool:
        """
        Create a collection if it doesn't exist
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the embedding vectors (default 384)
            distance: Distance metric (default COSINE)
            description: Description of the collection
            
        Returns:
            True if collection was created, False if it already existed
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if collection_name in collection_names:
                console.print(f"[yellow]⚠[/yellow] Collection '{collection_name}' already exists")
                return False
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            
            console.print(f"[green]✓[/green] Created collection '{collection_name}' ({description})")
            return True
            
        except Exception as e:
            console.print(f"[red]✗[/red] Error creating collection '{collection_name}': {str(e)}")
            raise
    
    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_type: PayloadSchemaType = PayloadSchemaType.KEYWORD
    ):
        """
        Create a payload index for efficient filtering
        
        Args:
            collection_name: Name of the collection
            field_name: Field name to index
            field_type: Type of the field
        """
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type
            )
            console.print(f"[green]✓[/green] Created index on '{collection_name}.{field_name}'")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Index creation warning for '{collection_name}.{field_name}': {str(e)}")
    
    def initialize_all_collections(self):
        """Initialize all collections with proper configuration"""
        
        console.print("\n[bold cyan]Initializing Qdrant Collections...[/bold cyan]\n")
        
        # 1. Mental Health Resources Collection (Global Search)
        self.create_collection_if_not_exists(
            collection_name=RESOURCES_COLLECTION,
            vector_size=384,
            distance=Distance.COSINE,
            description="Global mental health resources - searchable by all users"
        )
        # Create indexes for efficient filtering
        self.create_payload_index(RESOURCES_COLLECTION, "category", PayloadSchemaType.KEYWORD)
        self.create_payload_index(RESOURCES_COLLECTION, "resource_type", PayloadSchemaType.KEYWORD)
        self.create_payload_index(RESOURCES_COLLECTION, "difficulty", PayloadSchemaType.KEYWORD)
        self.create_payload_index(RESOURCES_COLLECTION, "tags", PayloadSchemaType.KEYWORD)
        
        # 2. User Memory Collection (User-specific)
        self.create_collection_if_not_exists(
            collection_name=MEMORY_COLLECTION,
            vector_size=384,
            distance=Distance.COSINE,
            description="User conversation memory - filtered by user_id"
        )
        # Create user_id index for filtering
        self.create_payload_index(MEMORY_COLLECTION, "user_id", PayloadSchemaType.KEYWORD)
        self.create_payload_index(MEMORY_COLLECTION, "session_id", PayloadSchemaType.KEYWORD)
        self.create_payload_index(MEMORY_COLLECTION, "mood", PayloadSchemaType.KEYWORD)
        
        # 3. User Profile Collection (User-specific)
        self.create_collection_if_not_exists(
            collection_name=USER_PROFILE_COLLECTION,
            vector_size=384,
            distance=Distance.COSINE,
            description="User profiles - filtered by user_id"
        )
        # Create user_id index for filtering
        self.create_payload_index(USER_PROFILE_COLLECTION, "user_id", PayloadSchemaType.KEYWORD)
        self.create_payload_index(USER_PROFILE_COLLECTION, "email", PayloadSchemaType.KEYWORD)
        
        # 4. Multimodal Data Collection (User-specific)
        self.create_collection_if_not_exists(
            collection_name=MULTIMODAL_COLLECTION,
            vector_size=384,
            distance=Distance.COSINE,
            description="Multimodal user data - filtered by user_id"
        )
        # Create user_id index for filtering
        self.create_payload_index(MULTIMODAL_COLLECTION, "user_id", PayloadSchemaType.KEYWORD)
        self.create_payload_index(MULTIMODAL_COLLECTION, "data_type", PayloadSchemaType.KEYWORD)
        
        console.print("\n[bold green]✓ Collection initialization complete![/bold green]\n")
        
        # Display collection info
        self.display_collections_info()
    
    def display_collections_info(self):
        """Display information about all collections"""
        console.print("[bold cyan]Collection Information:[/bold cyan]\n")
        
        collections_info = {
            RESOURCES_COLLECTION: "Global - Mental health resources accessible to all users",
            MEMORY_COLLECTION: "User-filtered - Conversation history and context",
            USER_PROFILE_COLLECTION: "User-filtered - User profiles and preferences",
            MULTIMODAL_COLLECTION: "User-filtered - Images, audio, and other multimodal data"
        }
        
        for collection_name, description in collections_info.items():
            try:
                info = self.client.get_collection(collection_name)
                console.print(f"[bold]{collection_name}[/bold]")
                console.print(f"  Description: {description}")
                console.print(f"  Points: {info.points_count}")
                console.print(f"  Vector Size: {info.config.params.vectors.size}")
                console.print(f"  Distance: {info.config.params.vectors.distance}")
                console.print()
            except Exception as e:
                console.print(f"[red]Error getting info for {collection_name}: {str(e)}[/red]\n")
    
    def delete_all_collections(self):
        """Delete all collections (use with caution!)"""
        console.print("[bold red]⚠ WARNING: Deleting all collections...[/bold red]\n")
        
        collections = [
            RESOURCES_COLLECTION,
            MEMORY_COLLECTION,
            USER_PROFILE_COLLECTION,
            MULTIMODAL_COLLECTION
        ]
        
        for collection_name in collections:
            try:
                self.client.delete_collection(collection_name)
                console.print(f"[red]✓[/red] Deleted collection '{collection_name}'")
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Could not delete '{collection_name}': {str(e)}")


def main():
    """Main initialization function"""
    try:
        # Initialize Qdrant
        initializer = QdrantInitializer()
        
        # Create all collections
        initializer.initialize_all_collections()
        
        console.print("[bold green]✓ Qdrant initialization successful![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Initialization failed: {str(e)}[/bold red]")
        raise


if __name__ == "__main__":
    # Set environment variables before running:
    # export QDRANT_URL="https://your-cluster.aws.cloud.qdrant.io:6333"
    # export QDRANT_API_KEY="your-api-key"
    
    main()
