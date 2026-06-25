# rag/seed_data.py
# Seed knowledge base with curated content for interviews and courses

from typing import List, Dict
from .rag_config import DocumentCategory


# =========================================================================
# INTERVIEW QUESTION EXAMPLES (by role/level)
# =========================================================================

INTERVIEW_QUESTIONS_DATA = [
    {
        "content": """Frontend Engineer Interview Question - Junior Level

Question: Explain the difference between var, let, and const in JavaScript.

Key Points to Cover:
- Scope: var is function-scoped, let/const are block-scoped
- Hoisting: var declarations are hoisted, let/const have temporal dead zone
- Reassignment: var and let allow reassignment, const does not
- Use Cases: Use const by default, let when reassignment needed, avoid var

Good Answer Framework:
1. Start with scope differences
2. Explain hoisting behavior
3. Discuss reassignment rules
4. Provide code examples
5. Recommend best practices""",
        "category": "interview_questions",
        "metadata": {
            "role": "frontend",
            "level": "junior",
            "skill_area": "javascript_fundamentals",
            "difficulty": "basic"
        }
    },
    {
        "content": """Backend Engineer Interview Question - Mid Level

Question: Design a system to handle user authentication and authorization.

Key Components:
- JWT tokens for stateless authentication
- Role-based access control (RBAC)
- Password hashing strategies
- Session management
- Token refresh mechanisms
- Rate limiting for security

Expected Solution Depth:
1. Database schema design for users/roles/permissions
2. Token generation and validation flow
3. Security considerations (CSRF, XSS, etc.)
4. Performance optimization
5. Scaling considerations for millions of users""",
        "category": "interview_questions",
        "metadata": {
            "role": "backend",
            "level": "mid",
            "skill_area": "system_design",
            "difficulty": "intermediate"
        }
    },
    {
        "content": """Data Engineer Interview Question - Senior Level

Question: Design a real-time data pipeline for processing streaming events from millions of devices.

Requirements & Approach:
- Handle high-throughput event streams (millions/sec)
- Ensure exactly-once processing semantics
- Build scalable distributed architecture
- Implement fault tolerance and recovery
- Optimize for low latency (<1 sec)

Technical Considerations:
1. Message broker selection (Kafka, Pub/Sub)
2. Stream processing framework (Spark, Flink)
3. State management and checkpointing
4. Monitoring and alerting
5. Cost optimization
6. Backup and disaster recovery""",
        "category": "interview_questions",
        "metadata": {
            "role": "data_engineer",
            "level": "senior",
            "skill_area": "distributed_systems",
            "difficulty": "advanced"
        }
    },
]


# =========================================================================
# EVALUATION RUBRICS
# =========================================================================

EVALUATION_RUBRICS_DATA = [
    {
        "content": """Code Quality and Architecture Evaluation Rubric

EXCELLENT (4/5):
- Clean, well-structured code
- Follows SOLID principles
- Proper error handling and edge cases
- Efficient algorithms and optimization
- Good naming conventions and documentation

GOOD (3/5):
- Code works correctly
- Generally follows best practices
- Minor optimization opportunities
- Adequate documentation
- Handles most edge cases

SATISFACTORY (2/5):
- Functional solution with issues
- Some code quality concerns
- Limited error handling
- Basic documentation
- Missing some edge cases

NEEDS IMPROVEMENT (1/5):
- Solution incomplete or incorrect
- Poor code structure
- Minimal error handling
- No documentation
- Doesn't handle edge cases""",
        "category": "evaluation_rubrics",
        "metadata": {
            "aspect": "code_quality",
            "applicable_to": ["technical_questions", "coding_challenges"],
            "severity": "high"
        }
    },
    {
        "content": """Communication and Problem-Solving Assessment

EXCELLENT:
- Articulates thinking clearly
- Explains approach before coding
- Asks clarifying questions
- Discusses trade-offs
- Communicates complexity fluently

GOOD:
- Generally clear communication
- Explains solution adequately
- Some questioning of requirements
- Mentions trade-offs
- Good pacing and clarity

SATISFACTORY:
- Basic explanation provided
- Communication somewhat unclear
- Limited questioning
- Little discussion of alternatives
- Could explain better

NEEDS IMPROVEMENT:
- Unclear or confusing explanations
- Doesn't ask clarifying questions
- No discussion of approach
- Rushed delivery
- Hard to follow thinking""",
        "category": "evaluation_rubrics",
        "metadata": {
            "aspect": "communication",
            "applicable_to": ["all_questions"],
            "severity": "high"
        }
    },
]


# =========================================================================
# SKILL FRAMEWORKS
# =========================================================================

SKILL_FRAMEWORKS_DATA = [
    {
        "content": """Frontend Engineering Skill Framework

JUNIOR FRONTEND ENGINEER:
- HTML/CSS fundamentals
- JavaScript ES6+ basics
- React fundamentals (components, props, state)
- Basic responsive design
- Git basics
- HTML forms and validation
- CSS flexbox and grid basics

SENIOR FRONTEND ENGINEER:
- Advanced React patterns (hooks, context, performance)
- TypeScript expertise
- State management (Redux, Zustand)
- Web performance optimization
- Accessibility (a11y) best practices
- Testing strategies (unit, integration, e2e)
- Build tools and tooling optimization
- Performance monitoring and debugging""",
        "category": "skill_frameworks",
        "metadata": {
            "role": "frontend",
            "focus": "skill_progression"
        }
    },
    {
        "content": """Backend Engineering Skill Framework

JUNIOR BACKEND ENGINEER:
- API design and REST principles
- Database basics (SQL, normalization)
- Authentication fundamentals
- Error handling and logging
- Testing basics
- Deployment basics
- Framework knowledge (Django, Flask, etc.)

SENIOR BACKEND ENGINEER:
- Microservices architecture
- Database optimization and scaling
- Caching strategies (Redis, memcached)
- Message queues and async processing
- Monitoring and observability
- Security best practices
- Performance tuning
- Infrastructure as Code (IaC)""",
        "category": "skill_frameworks",
        "metadata": {
            "role": "backend",
            "focus": "skill_progression"
        }
    },
]


# =========================================================================
# BEST PRACTICES
# =========================================================================

BEST_PRACTICES_DATA = [
    {
        "content": """Interview Answer Best Practices

1. STRUCTURE YOUR ANSWER:
   - Clarify requirements (ask questions!)
   - Outline your approach at high level
   - Implement step by step
   - Test and validate
   - Discuss optimization opportunities

2. COMMUNICATION TIPS:
   - Think out loud
   - Explain your reasoning
   - Acknowledge trade-offs
   - Admit when you're unsure
   - Ask for feedback

3. COMMON MISTAKES TO AVOID:
   - Jumping into code without planning
   - Ignoring edge cases
   - Writing without explanation
   - Being defensive about feedback
   - Rushing through the solution

4. TIME MANAGEMENT:
   - Allocate 10% for requirements
   - 20% for design discussion
   - 40% for implementation
   - 20% for testing
   - 10% for optimization""",
        "category": "best_practices",
        "metadata": {
            "aspect": "interview_technique",
            "applicable_to": ["all_interview_types"]
        }
    },
    {
        "content": """System Design Answer Best Practices

1. APPROACH:
   - Clarify requirements and constraints
   - Define high-level components
   - Discuss data flow
   - Address scalability
   - Consider failure scenarios

2. KEY COMPONENTS:
   - Load balancing
   - Caching layers
   - Database sharding
   - Message queues
   - Monitoring/logging

3. EVALUATION CRITERIA:
   - Handles scale requirements
   - Addresses single points of failure
   - Optimizes for latency
   - Cost-efficient
   - Operationally viable

4. RED FLAGS TO AVOID:
   - Oversimplified architecture
   - No caching strategy
   - No failure handling
   - Unclear communication
   - Unfeasible solutions""",
        "category": "best_practices",
        "metadata": {
            "aspect": "system_design",
            "applicable_to": ["system_design_questions"]
        }
    },
]


# =========================================================================
# TECHNICAL CONTENT
# =========================================================================

TECHNICAL_CONTENT_DATA = [
    {
        "content": """Distributed Systems Fundamentals

KEY CONCEPTS:
1. Scalability
   - Horizontal scaling (add more machines)
   - Vertical scaling (bigger machines)
   - Load balancing
   - Partitioning/Sharding

2. Reliability
   - Fault tolerance
   - Replication strategies
   - Consistency models (strong, eventual)
   - Consensus algorithms (Raft, Paxos)

3. Performance
   - Latency optimization
   - Throughput maximization
   - Caching strategies
   - Index optimization

4. Common Patterns
   - Master-slave replication
   - Leader election
   - Distributed consensus
   - Circuit breaker pattern""",
        "category": "technical_content",
        "metadata": {
            "topic": "distributed_systems",
            "level": "intermediate"
        }
    },
    {
        "content": """Database Design and Optimization

RELATIONAL DATABASES:
- ACID properties (Atomicity, Consistency, Isolation, Durability)
- Normalization (1NF, 2NF, 3NF, BCNF)
- Indexing strategies (B-tree, hash, bitmap)
- Query optimization

NOSQL DATABASES:
- Document stores (MongoDB)
- Key-value stores (Redis, DynamoDB)
- Graph databases (Neo4j)
- CAP theorem implications

OPTIMIZATION TECHNIQUES:
- Connection pooling
- Query caching
- Database sharding
- Read replicas
- Materialized views""",
        "category": "technical_content",
        "metadata": {
            "topic": "database_design",
            "level": "intermediate"
        }
    },
]


# =========================================================================
# COMMUNICATION CONTENT
# =========================================================================

COMMUNICATION_CONTENT_DATA = [
    {
        "content": """Effective Technical Communication in Interviews

SPEAKING CLEARLY:
- Speak at moderate pace
- Use simple language for complex ideas
- Avoid filler words (um, uh, like)
- Take pauses to think
- Articulate edge cases explicitly

EXPLAINING YOUR THINKING:
- Start with "I'm thinking..."
- Break down complex problems
- Provide examples
- Draw diagrams/pseudocode
- Summarize key points

ASKING QUESTIONS:
- Clarify ambiguous requirements
- Understand constraints
- Ask for feedback: "Does this approach make sense?"
- Explore alternatives: "What if we tried...?"

HANDLING FEEDBACK:
- Listen completely before responding
- Thank interviewer for feedback
- Adapt quickly and positively
- Show flexibility and learning mindset""",
        "category": "communication_content",
        "metadata": {
            "aspect": "technical_communication",
            "applicable_to": ["interviews", "presentations"]
        }
    },
]


# =========================================================================
# COURSE MATERIALS
# =========================================================================

COURSE_MATERIALS_DATA = [
    {
        "content": """JavaScript Advanced Patterns - Course Module

LEARNING OBJECTIVES:
- Master closure and scope
- Understand prototypal inheritance
- Learn functional programming concepts
- Implement design patterns
- Handle asynchronous code

TOPICS:
1. Closures and Scope
   - Function scope
   - Block scope (let, const)
   - Closure patterns
   - Module pattern

2. Prototypal Inheritance
   - Prototype chain
   - Object.create()
   - Constructor functions
   - Class syntax

3. Functional Programming
   - Pure functions
   - Higher-order functions
   - Array methods (map, filter, reduce)
   - Composition and currying

4. Design Patterns
   - Observer pattern
   - Singleton pattern
   - Factory pattern
   - Middleware pattern""",
        "category": "course_materials",
        "metadata": {
            "course_name": "JavaScript Advanced",
            "module": "patterns_and_concepts",
            "level": "intermediate"
        }
    },
    {
        "content": """System Design Course - Scalability Module

LEARNING OBJECTIVES:
- Design scalable systems
- Choose appropriate architectures
- Handle millions of users
- Optimize for performance

TOPICS:
1. Load Balancing
   - Round-robin
   - Least connections
   - Geographic distribution

2. Caching Strategies
   - Client-side caching
   - Server-side caching
   - Cache invalidation
   - Distributed caching

3. Database Scaling
   - Replication
   - Sharding strategies
   - Master-slave architecture
   - CQRS pattern

4. Microservices
   - Service decomposition
   - API gateways
   - Service discovery
   - Inter-service communication""",
        "category": "course_materials",
        "metadata": {
            "course_name": "System Design",
            "module": "scalability",
            "level": "advanced"
        }
    },
]


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def get_seed_data_by_category(category: DocumentCategory) -> List[Dict]:
    """Get seed data for a specific category."""
    
    category_map = {
        DocumentCategory.INTERVIEW_QUESTIONS: INTERVIEW_QUESTIONS_DATA,
        DocumentCategory.EVALUATION_RUBRICS: EVALUATION_RUBRICS_DATA,
        DocumentCategory.SKILL_FRAMEWORKS: SKILL_FRAMEWORKS_DATA,
        DocumentCategory.BEST_PRACTICES: BEST_PRACTICES_DATA,
        DocumentCategory.TECHNICAL_CONTENT: TECHNICAL_CONTENT_DATA,
        DocumentCategory.COMMUNICATION_CONTENT: COMMUNICATION_CONTENT_DATA,
        DocumentCategory.COURSE_MATERIALS: COURSE_MATERIALS_DATA,
    }
    
    return category_map.get(category, [])


def get_all_seed_data() -> List[Dict]:
    """Get all seed data."""
    return (
        INTERVIEW_QUESTIONS_DATA +
        EVALUATION_RUBRICS_DATA +
        SKILL_FRAMEWORKS_DATA +
        BEST_PRACTICES_DATA +
        TECHNICAL_CONTENT_DATA +
        COMMUNICATION_CONTENT_DATA +
        COURSE_MATERIALS_DATA
    )


def format_seed_data_for_ingestion() -> List[Dict]:
    """Format seed data for document ingestion."""
    
    all_data = []
    
    for category in DocumentCategory:
        seed_items = get_seed_data_by_category(category)
        
        for i, item in enumerate(seed_items):
            doc = {
                "content": item["content"],
                "category": category.value,
                "metadata": item.get("metadata", {}),
                "source_id": f"{category.value}_{i}"
            }
            all_data.append(doc)
    
    return all_data
