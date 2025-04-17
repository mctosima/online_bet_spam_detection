# Lightweight Transformer Architecture

The following diagram illustrates the architecture of the LightweightTransformer model for text classification.

```mermaid
graph TD
    A[Input Token IDs] --> B[Token Embeddings]
    A --> M[Attention Mask]
    B --> C[Add Positional Embeddings]
    C --> D[Dropout]
    
    subgraph "Transformer Blocks x N"
        D --> E["Self-Attention<br>MultiHeadSelfAttention"]
        E --> F[Add & Norm]
        F --> G[Feed Forward Network]
        G --> H[Add & Norm]
    end
    
    H --> I["Extract [CLS] Token<br>First Token Representation"]
    I --> J[Layer Normalization]
    J --> K[Dropout]
    K --> L[Linear Classification Head]
    
    subgraph "MultiHeadSelfAttention"
        N[Input] --> O["Q, K, V Projections"]
        O --> P["Scaled Dot-Product Attention"]
        M -.-> P
        P --> Q["Concat Heads"]
        Q --> R["Output Projection"]
    end
    
    style A fill:#d9f7be,stroke:#389e0d,stroke-width:2px
    style B fill:#fff2e8,stroke:#d4380d,stroke-width:2px
    style C fill:#fff2e8,stroke:#d4380d,stroke-width:2px
    style D fill:#f4f4f4,stroke:#434343,stroke-width:1px
    style E fill:#d6e4ff,stroke:#1d39c4,stroke-width:2px
    style F fill:#f9f0ff,stroke:#722ed1,stroke-width:2px
    style G fill:#d6e4ff,stroke:#1d39c4,stroke-width:2px
    style H fill:#f9f0ff,stroke:#722ed1,stroke-width:2px
    style I fill:#fff2e8,stroke:#d4380d,stroke-width:2px
    style J fill:#f9f0ff,stroke:#722ed1,stroke-width:2px
    style K fill:#f4f4f4,stroke:#434343,stroke-width:1px
    style L fill:#f6ffed,stroke:#52c41a,stroke-width:2px
    style M fill:#e6f7ff,stroke:#096dd9,stroke-width:1px
    style N fill:#f4f4f4,stroke:#434343,stroke-width:1px
    style O fill:#fff2e8,stroke:#d4380d,stroke-width:1px
    style P fill:#d6e4ff,stroke:#1d39c4,stroke-width:1px
    style Q fill:#f9f0ff,stroke:#722ed1,stroke-width:1px
    style R fill:#e6f7ff,stroke:#096dd9,stroke-width:1px
```

## Model Description

The LightweightTransformer is a simplified version of transformer architecture designed for text classification:

1. **Input Processing**:
   - Token IDs are converted to embeddings
   - Positional information is added via learned positional embeddings

2. **Transformer Encoder** (repeated N times):
   - **Self-Attention Block**: Multi-head self-attention mechanism
   - **Feed-Forward Network**: Two linear transformations with GELU activation

3. **Classification**:
   - Extract the first token ([CLS]) representation
   - Process through normalization, dropout, and linear classification layer

## Model Parameters
- Vocabulary Size: 30,522 (Default for BERT tokenizer)
- Embedding Dimension: 256
- Number of Heads: 4
- Feed-Forward Dimension: 512
- Number of Layers: 3
- Default Maximum Sequence Length: 128
- Default Dropout Rate: 0.1

This architecture provides a more efficient alternative to full-scale transformer models like BERT, with significantly fewer parameters while maintaining strong performance for text classification tasks.
