using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Sessions;
using Xunit;

namespace OpenTail.Stingray.Tests.Sessions;

public class SkillCompositionTests
{
    [Fact]
    public void ISkill_CanBeConstructedAndAttached()
    {
        var skill = EchoSkill.Create();

        Assert.Equal("echo", skill.Name);
        Assert.Single(skill.Instructions);
        Assert.Single(skill.Tools);
        Assert.Single(skill.Resources);
        Assert.Equal("echo_value", skill.Tools[0].Name);
    }

    [Fact]
    public void AttachSkill_StoresSkillAndValidatesTools()
    {
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 16);
        var session = new InferenceSession(cache);
        var skill = EchoSkill.Create();

        session.AttachSkill(skill);

        Assert.Single(session.AttachedSkills);
        Assert.Equal("echo", session.AttachedSkills[0].Name);

        var toolCall = new OpenTail.Stingray.Core.Tools.ToolCall("call_1", "echo_value", default);
        Assert.True(session.ValidateToolCall(toolCall));

        var unknownCall = new OpenTail.Stingray.Core.Tools.ToolCall("call_2", "unknown_tool", default);
        Assert.False(session.ValidateToolCall(unknownCall));
    }

    [Fact]
    public void DetachSkill_RemovesSkillFromSession()
    {
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 16);
        var session = new InferenceSession(cache);
        var skill = EchoSkill.Create();

        session.AttachSkill(skill);
        Assert.Single(session.AttachedSkills);

        bool removed = session.DetachSkill("echo");

        Assert.True(removed);
        Assert.Empty(session.AttachedSkills);

        var toolCall = new OpenTail.Stingray.Core.Tools.ToolCall("call_1", "echo_value", default);
        Assert.False(session.ValidateToolCall(toolCall));
    }

    [Fact]
    public void AttachSkill_RejectsDuplicateToolNames()
    {
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 16);
        var session = new InferenceSession(cache);
        var skill1 = new Skill("skill1", Tools: new[] { new Tool("shared_tool") });
        var skill2 = new Skill("skill2", Tools: new[] { new Tool("shared_tool") });

        session.AttachSkill(skill1);

        var ex = Assert.Throws<InvalidOperationException>(() => session.AttachSkill(skill2));
        Assert.Contains("Tool name collision", ex.Message);
    }

    [Fact]
    public void Fork_InheritsAttachedSkillsInIsolation()
    {
        using var cache = new CpuKvCache(totalPages: 50, pageSizeTokens: 16);
        var parent = new InferenceSession(cache);
        var skill = EchoSkill.Create();
        parent.AttachSkill(skill);

        var fork = parent.Fork();

        Assert.Single(fork.AttachedSkills);
        Assert.Equal("echo", fork.AttachedSkills[0].Name);

        // Detaching on child must not affect parent
        fork.DetachSkill("echo");

        Assert.Empty(fork.AttachedSkills);
        Assert.Single(parent.AttachedSkills);
    }
}
