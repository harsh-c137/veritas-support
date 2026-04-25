import json
import os

def generate_blogs():
    os.makedirs("../data", exist_ok=True)
    
    blogs_data = [
        {
            "content": "Hot take: The official Nexus Agent docs are wrong about hallucination fixes. They tell you to drop the temperature to 0.1, but I've found that makes the agent sound like a robot. Instead, keep the temperature at 0.5 but inject a strict 'SYSTEM OVERRIDE' prompt before every query. It works 10x better for our internal tools.",
            "metadata": {
                "source_type": "blog",
                "title": "Stop crippling your Nexus Agent's temperature",
                "author": "DevOpsDan",
                "date_published": "2026-03-12",
                "url": "https://devopsdan.io/nexus-temperature-hack"
            }
        },
        {
            "content": "We migrated our customer support widget to Nexus Agent last week. Overall, the TypeScript types are solid, but watch out for the `<NexusProvider>`. If you are using React 18, strict mode causes the provider to mount twice, which occasionally drops the API key from context. The quick fix is to memoize the API key prop before passing it in.",
            "metadata": {
                "source_type": "blog",
                "title": "Surviving the Nexus Agent React Migration",
                "author": "Sarah Codes",
                "date_published": "2026-04-01",
                "url": "https://sarahcodes.dev/nexus-react-quirks"
            }
        },
        {
            "content": "Everyone is obsessing over the retrieval engine, but nobody is talking about the AWS costs. If you blindly run `nexus-cli init`, it provisions a dedicated Route53 hosted zone. If you are just testing, do NOT do this. Manually configure the SES sandbox first, or you will wake up to a $50 surprise bill.",
            "metadata": {
                "source_type": "blog",
                "title": "Hidden AWS Costs in Nexus Agent Deployments",
                "author": "Cloud Optimizer",
                "date_published": "2026-02-28",
                "url": "https://cloudoptimizer.net/nexus-billing-trap"
            }
        }
    ]

    output_path = "../data/blogs.json"
    with open(output_path, "w") as f:
        json.dump(blogs_data, f, indent=4)
        
    print(f"✅ Successfully generated {len(blogs_data)} blog chunks at {output_path}")

if __name__ == "__main__":
    generate_blogs()