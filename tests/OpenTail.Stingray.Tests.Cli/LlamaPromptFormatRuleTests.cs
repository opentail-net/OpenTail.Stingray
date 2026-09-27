using OpenTail.Stingray.Cli;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class LlamaPromptFormatRuleTests
{
    private static void ResetState()
    {
        RunCommand.s_arch = "llama";
        RunCommand.s_jinja = null;
        RunCommand.s_hasLlama3Headers = false;
    }

    [Fact]
    public void FormatPrompt_LlamaArch_NoLlama3Headers_RendersVicuna()
    {
        ResetState();
        try
        {
            RunCommand.s_arch = "llama";
            RunCommand.s_hasLlama3Headers = false;

            string prompt = RunCommand.FormatPrompt("Hello world", null);
            Assert.Equal("USER: Hello world\nASSISTANT:", prompt);
        }
        finally
        {
            ResetState();
        }
    }

    [Fact]
    public void FormatPrompt_LlamaArch_Llama3Headers_RendersLlama3Headers()
    {
        ResetState();
        try
        {
            RunCommand.s_arch = "llama";
            RunCommand.s_hasLlama3Headers = true;

            string prompt = RunCommand.FormatPrompt("Hello world", null);
            Assert.StartsWith("<|begin_of_text|>", prompt);
            Assert.Contains("<|start_header_id|>user<|end_header_id|>", prompt);
            Assert.Contains("<|start_header_id|>assistant<|end_header_id|>", prompt);
        }
        finally
        {
            ResetState();
        }
    }

    [Fact]
    public void FormatPrompt_LlamaArch_Llama3Headers_WithSystemPrompt_RendersSystemHeader()
    {
        ResetState();
        try
        {
            RunCommand.s_arch = "llama";
            RunCommand.s_hasLlama3Headers = true;

            string prompt = RunCommand.FormatPrompt("Hello world", "You are an assistant");
            Assert.StartsWith("<|begin_of_text|>", prompt);
            Assert.Contains("<|start_header_id|>system<|end_header_id|>\n\nYou are an assistant<|eot_id|>", prompt);
            Assert.Contains("<|start_header_id|>user<|end_header_id|>\n\nHello world<|eot_id|>", prompt);
            Assert.Contains("<|start_header_id|>assistant<|end_header_id|>", prompt);
        }
        finally
        {
            ResetState();
        }
    }

    [Fact]
    public void FormatPrompt_JinjaTemplatePresent_OverridesHardcodedRules()
    {
        ResetState();
        try
        {
            RunCommand.s_arch = "llama";
            RunCommand.s_hasLlama3Headers = true;
            RunCommand.s_jinja = new JinjaChatTemplate("CUSTOM_JINJA: {{ messages[0]['content'] }}");

            string prompt = RunCommand.FormatPrompt("Hello world", null);
            Assert.Equal("CUSTOM_JINJA: Hello world", prompt);
        }
        finally
        {
            ResetState();
        }
    }
}
